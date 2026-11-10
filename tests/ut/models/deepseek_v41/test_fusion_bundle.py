# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import pytest

FUSION_DIR = Path(__file__).resolve().parents[4] / "examples/deepseek_v41_a5/fusion"


class _NormalizeDelete(ast.NodeTransformer):
    """Ruff can spell del a, b as del (a, b); both delete in the same order."""

    def visit_Delete(self, node):
        def unpack(items):
            return [child for item in items for child in (unpack(item.elts) if isinstance(item, ast.Tuple) else [item])]

        node.targets = unpack(node.targets)
        return node


@pytest.mark.parametrize("profile", ["selected", "optional"])
def test_packaged_fusion_callbacks_match_measured_asts(tmp_path, profile):
    prepare = runpy.run_path(str(FUSION_DIR / "prepare-bundle.py"))["prepare_bundle"]
    output = tmp_path / "editable"
    manifest = prepare(output, profile)
    provenance = json.loads((FUSION_DIR / "provenance.json").read_text())["profiles"][profile]
    for name in manifest["files"]:
        tree = _NormalizeDelete().visit(ast.parse((output / name).read_text()))
        # Python 3.12 adds empty type_params to functions parsed on Python 3.11.
        # Exclude only that empty metadata; nonempty generic parameters remain.
        for node in ast.walk(tree):
            if "type_params" in node._fields and not node.type_params:
                node._fields = tuple(field for field in node._fields if field != "type_params")
        digest = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
        assert digest == provenance["files"][name]["ast_sha256"], name
    for overrides in manifest["method_overrides"].values():
        for override in overrides:
            assert override["file"] in manifest["files"]
    with pytest.raises(FileExistsError):
        prepare(output, profile)


def test_packaged_fusion_rejects_corruption_before_creating_output(tmp_path):
    prepare = runpy.run_path(str(FUSION_DIR / "prepare-bundle.py"))["prepare_bundle"]
    source = tmp_path / "source"
    shutil.copytree(FUSION_DIR / "selected", source / "selected")
    callback = source / "selected" / "cache.py"
    callback.write_text(callback.read_text() + "\n# unexpected edit\n")
    output = tmp_path / "editable"
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare(output, root=source)
    assert not output.exists()


def test_packaged_fusion_rejects_different_checkpoint(tmp_path):
    prepare = runpy.run_path(str(FUSION_DIR / "prepare-bundle.py"))["prepare_bundle"]
    config = tmp_path / "config.json"
    config.write_text("{}")
    output = tmp_path / "editable"
    with pytest.raises(ValueError, match="Checkpoint config differs"):
        prepare(output, config=config)
    assert not output.exists()
