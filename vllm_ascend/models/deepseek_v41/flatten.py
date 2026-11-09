# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Source-derived inference backbone inlining, without changing device ops.

Inspired by Agentic-Compiler's inline/verify stages. Attention, Engram, MoE,
quantized projections and serving state remain explicit effectful boundaries.
This is deliberately not the training compiler's pure-Torch normal form.
"""

import ast
import copy
import hashlib
import inspect
import json
import textwrap
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from . import core


def _parse(function):
    return ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]


def _expr(source):
    return ast.parse(source, mode="eval").body


def _assign(target, value):
    target = copy.deepcopy(target)
    for node in ast.walk(target):
        if isinstance(node, (ast.Name, ast.Tuple, ast.List)):
            node.ctx = ast.Store()
    return ast.Assign(targets=[target], value=value)


def _delete(names):
    return ast.Delete(targets=[ast.Name(id=name, ctx=ast.Del()) for name in sorted(set(names))])


class _Substitute(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping

    def visit_Name(self, node):
        if node.id not in self.mapping:
            return node
        result = copy.deepcopy(self.mapping[node.id])
        if isinstance(result, ast.Name):
            result.ctx = node.ctx
        return result


class _CoreInliner:
    """Inline a deliberately restricted set of single-exit tensor primitives."""

    def __init__(self, config):
        self.config = config
        self.counter = 0
        self.functions = {
            name: _parse(getattr(core, name))
            for name in ("rms_norm", "hc_mixes", "hc_split_sinkhorn", "hc_pre", "hc_post")
        }

    def expression(self, node):
        before, cleanup = [], []
        node = copy.deepcopy(node)
        # These primitives have no calls in short-circuit/conditional expressions.
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.expr):
                pre, replacement, post = self.expression(value)
                before.extend(pre)
                cleanup.extend(post)
                setattr(node, field, replacement)
            elif isinstance(value, list):
                updated = []
                for item in value:
                    if isinstance(item, ast.expr):
                        pre, item, post = self.expression(item)
                        before.extend(pre)
                        cleanup.extend(post)
                    updated.append(item)
                setattr(node, field, updated)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.functions:
            if node.keywords:
                raise ValueError("Core inlining expects explicit positional arguments")
            pre, result = self.call(node.func.id, node.args)
            return before + pre, result, cleanup + [result.id]
        return before, node, cleanup

    def statements(self, statements):
        result = []
        for statement in statements:
            statement = copy.deepcopy(statement)
            if isinstance(statement, ast.For):
                # Only the Sinkhorn loop enters this pass; specialize its bound.
                if not isinstance(statement.iter, ast.Call) or ast.unparse(statement.iter.func) != "range":
                    raise ValueError("Unexpected loop in core primitive")
                bound = statement.iter.args[0]
                if not (
                    isinstance(bound, ast.BinOp)
                    and isinstance(bound.op, ast.Sub)
                    and isinstance(bound.left, ast.Constant)
                    and isinstance(bound.right, ast.Constant)
                ):
                    raise ValueError("Sinkhorn iteration count must be static")
                for _ in range(bound.left.value - bound.right.value):
                    result.extend(self.statements(statement.body))
            elif isinstance(statement, ast.If):
                statement.body = self.statements(statement.body)
                statement.orelse = self.statements(statement.orelse)
                result.append(statement)
            elif isinstance(statement, ast.Assign):
                pre, statement.value, cleanup = self.expression(statement.value)
                result.extend(pre)
                result.append(statement)
                if cleanup:
                    result.append(_delete(cleanup))
            else:
                result.append(statement)
        return result

    def call(self, name, args):
        function = copy.deepcopy(self.functions[name])
        parameters = [argument.arg for argument in function.args.args]
        defaults = dict(zip(parameters[-len(function.args.defaults) :], function.args.defaults))
        arguments = dict(zip(parameters, args))
        arguments = {parameter: arguments.get(parameter, defaults.get(parameter)) for parameter in parameters}
        if any(value is None for value in arguments.values()):
            raise ValueError(f"Missing argument for {name}")
        if name == "hc_split_sinkhorn":
            arguments["hc_mult"] = ast.Constant(self.config.hc_mult)
        self.counter += 1
        prefix = f"inline_{self.counter}_"
        locals_ = {
            node.id
            for node in ast.walk(function)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id != "_"
        }
        mapping = {local: ast.Name(id=prefix + local, ctx=ast.Load()) for local in locals_}
        bindings, bound_names = [], []
        for parameter, value in arguments.items():
            if isinstance(value, ast.Constant):
                mapping[parameter] = value
            else:
                target = ast.Name(id=prefix + parameter, ctx=ast.Load())
                mapping[parameter] = target
                bindings.append(_assign(target, value))
                bound_names.append(target.id)
        body = [_Substitute(mapping).visit(statement) for statement in function.body]
        if not isinstance(body[-1], ast.Return):
            raise ValueError("Core primitive must have a final return")
        result = ast.Name(id=prefix + "result", ctx=ast.Load())
        body[-1] = _assign(result, body[-1].value)
        body = self.statements(body)
        # Preserve function-scope tensor lifetimes: flattening must not retain
        # every layer's temporaries until the entire backbone returns.
        cleanup = bound_names + [prefix + local for local in locals_]
        return bindings + body + [_delete(cleanup)], result


class _BlockCalls(ast.NodeTransformer):
    def __init__(self, config):
        self.config = config

    def visit_Call(self, node):
        node = self.generic_visit(node)
        name = ast.unparse(node.func)
        if name == "self.mixes":
            part = ast.literal_eval(node.args[1])
            return ast.Call(
                func=_expr("hc_mixes"),
                args=[
                    node.args[0],
                    _expr(f"self.hc_{part}_fn"),
                    _expr(f"self.hc_{part}_scale"),
                    _expr(f"self.hc_{part}_base"),
                    ast.Constant(self.config.rms_norm_eps),
                    ast.Constant(self.config.hc_eps),
                    ast.Constant(self.config.hc_sinkhorn_iters),
                ],
                keywords=[],
            )
        if name in ("self.attn_norm", "self.ffn_norm"):
            return ast.Call(
                func=_expr("rms_norm"),
                args=[node.args[0], _expr(name + ".weight"), ast.Constant(self.config.rms_norm_eps)],
                keywords=[],
            )
        return node


def _block_body(block_forward, config, call, target, inliner):
    function = _BlockCalls(config).visit(_parse(block_forward))
    parameters = [argument.arg for argument in function.args.args]
    arguments = {"self": call.func, "decode_mask": ast.Constant(None), "image_mask": ast.Constant(None)}
    arguments.update(zip(parameters[1:], call.args))
    arguments.update({keyword.arg: keyword.value for keyword in call.keywords})
    # Reuse these names across statically expanded blocks; each block drops them.
    local_names = {
        node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    mapping = {name: _expr("block_" + name) for name in local_names | set(parameters)}
    bindings = [_assign(mapping[name], value) for name, value in arguments.items()]
    body = [_Substitute(mapping).visit(statement) for statement in function.body]
    empty_branch = body[0]
    if not isinstance(empty_branch, ast.If) or not isinstance(empty_branch.body[-1], ast.Return):
        raise ValueError("Block empty-token guard changed; review the flattening pass")
    empty_branch.body[-1] = _assign(target, empty_branch.body[-1].value)
    if not isinstance(body[-1], ast.Return):
        raise ValueError("Block return structure changed")
    body[-1] = _assign(target, body[-1].value)
    nonempty = inliner.statements(body[1:])
    empty_branch.orelse = nonempty + [_delete(["block_" + name for name in local_names - set(parameters)])]
    return bindings + [empty_branch, _delete(["block_" + name for name in arguments])]


def generate_forward(function, block_forward, config, layer_count, *, draft=False):
    """Return reviewable source with static layers and HC/core calls expanded."""
    if layer_count < 1 or config.hc_sinkhorn_iters < 1 or config.hc_mult < 1:
        raise ValueError("Layer count, HC width and Sinkhorn iterations must be positive")
    tree = _parse(function)
    tree.decorator_list = []
    inliner = _CoreInliner(config)

    def rewrite(statements):
        output = []
        for statement in statements:
            statement = copy.deepcopy(statement)
            if isinstance(statement, ast.For) and ast.unparse(statement.iter) in (
                "enumerate(self.layers)",
                "self.mtp",
            ):
                for layer_id in range(layer_count):
                    mapping = (
                        {"stage": _expr(f"self.mtp[{layer_id}]")}
                        if draft
                        else {"layer": _expr(f"self.layers[{layer_id}]"), "layer_id": ast.Constant(layer_id)}
                    )
                    output.extend(rewrite([_Substitute(mapping).visit(copy.deepcopy(item)) for item in statement.body]))
            elif isinstance(statement, ast.If):
                statement.body = rewrite(statement.body)
                statement.orelse = rewrite(statement.orelse)
                output.append(statement)
            elif (
                isinstance(statement, ast.Assign)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Subscript)
                and ast.unparse(statement.value.func.value) in ("self.layers", "self.mtp")
            ):
                output.extend(_block_body(block_forward, config, statement.value, statement.targets[0], inliner))
            else:
                output.append(statement)
        return output

    tree.body = rewrite(tree.body)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def specialization(config):
    """Configuration constants baked into this pass, including loop extents."""
    return {
        name: getattr(config, name)
        for name in ("num_hidden_layers", "hc_mult", "hc_sinkhorn_iters", "rms_norm_eps", "hc_eps")
    } | {"draft_layers": len(config.compress_ratios) - config.num_hidden_layers}


def load_forward_source(directory, config, name, *, draft=False):
    """Load an explicitly selected, integrity-checked editable source bundle."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("specialization") != specialization(config):
        raise ValueError("Flattened bundle specialization does not match this checkpoint")
    filename = ("draft_" if draft else "target_") + name + ".py"
    return _load_bundle_file(directory, manifest, filename)


def _load_bundle_file(directory, manifest, filename):
    if Path(filename).name != filename or not filename.endswith(".py"):
        raise ValueError("Flattened source files must be Python files within the bundle")
    source = (directory / filename).read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if manifest["files"][filename]["sha256"] != digest:
        raise ValueError(f"Flattened source checksum mismatch: {filename}; record edits in a new experiment manifest")
    return source, str(directory / filename)


def install_leaf_overrides(model, directory, *, draft=False):
    """Expose selected effectful boundaries as explicit per-instance callbacks.

    An override never changes a module class or global function used by another
    model. Each callback keeps the globals of its original method, plus the
    imports/helpers explicitly supplied by its checksum-checked source file.
    """
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("specialization") != specialization(model.config):
        raise ValueError("Flattened bundle specialization does not match this checkpoint")
    entries = manifest.get("method_overrides", {}).get("draft" if draft else "target", [])
    loaded = {}
    for entry in entries:
        targets = [model]
        for component in entry["path"].split("."):
            if component == "*":
                targets = [child for target in targets for child in target]
            else:
                targets = [getattr(target, component) for target in targets]
        source, filename = _load_bundle_file(directory, manifest, entry["file"])
        code = compile(source, filename, "exec")
        for target in targets:
            for method in entry["methods"]:
                original = getattr(type(target), method)
                namespace = dict(original.__globals__, torch=torch, F=F)
                exec(code, namespace)
                if not inspect.isfunction(namespace.get(method)):
                    raise ValueError(f"Flattened callback file does not define {method}")
                setattr(target, method, MethodType(namespace[method], target))
        loaded[entry["file"]] = source
    model.flattened_leaf_sources = loaded


def install_flattened_forward(model, block_forward, *, draft=False, source_dir=None):
    """Bind only this instance; parameter ownership and state_dict stay intact."""
    names = ("forward",) if draft else ("forward", "_forward_decode")
    layer_count = len(model.mtp if draft else model.layers)
    sources = {}
    for name in names:
        original = getattr(type(model), name)
        if source_dir is None:
            source = generate_forward(original, block_forward, model.config, layer_count, draft=draft)
            filename = f"<v41-flat-{name}>"
        else:
            source, filename = load_forward_source(source_dir, model.config, name, draft=draft)
        namespace = dict(original.__globals__, torch=torch, F=F)
        exec(compile(source, filename, "exec"), namespace)
        setattr(model, name, MethodType(namespace[name], model))
        sources[name] = source
    model.flattened_forward_sources = sources
    if source_dir is not None:
        install_leaf_overrides(model, source_dir, draft=draft)
