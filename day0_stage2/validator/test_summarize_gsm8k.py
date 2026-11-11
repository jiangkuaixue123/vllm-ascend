# SPDX-License-Identifier: Apache-2.0
"""CPU-only fixtures for the AISBench receipt join. They contain no benchmark examples."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_gsm8k import audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_lines(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


class TestSummarizeGsm8k(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.run = self.base / "run"
        self.data = self.base / "data"
        self.source = self.base / "source"
        self.inputs = self.base / "inputs.json"
        self.gates = self.base / "gates.json"
        self.config = self.run / "work/fixture/configs/resolved.py"
        self.predfile = self.run / "work/fixture/predictions/test-model/gsm8k.jsonl"
        self.official = self.run / "work/fixture/results/test-model/gsm8k.json"
        questions = ["fixture-a", "fixture-b", "fixture-c"]
        self.ids = [sha(q.encode()) for q in questions]
        samples = [{"question": q, "answer": f"fixture #### {i}"} for i, q in enumerate(questions)]
        write_lines(self.data / "test.jsonl", samples)
        template = [{"role": role, "prompt": f"demo-{i}"} for i, role in enumerate(["HUMAN", "BOT"] * 4)]
        template.append({"role": "HUMAN", "prompt": "Question: {question}\nLet's think step by step\nAnswer:"})
        promptfile = self.source / "configs/test_dataset.py"
        promptfile.parent.mkdir(parents=True)
        promptfile.write_text(f"gsm8k_infer_cfg=dict(prompt_template=dict(template=dict(round={template!r})))\n")
        inputs = dict(
            scoring_sources={"configs/test_dataset.py": sha(promptfile.read_bytes())},
            ordered_ids=self.ids,
            inputs={"test.jsonl": sha((self.data / "test.jsonl").read_bytes())},
            reader_test_range="[:3]",
            aisbench_revision="fixture-framework",
            dataset_revision="fixture-data",
        )
        write(self.inputs, inputs)
        gen = dict(
            chat_template_kwargs=dict(reasoning_effort="high", thinking_mode="chat"), ignore_eos=False, temperature=0
        )
        benchmark = dict(
            model="test_model",
            dataset="test_dataset",
            count=3,
            reader_range="[:3]",
            client_max_inflight=2,
            max_http_attempts_per_case=1,
            max_output_tokens=20,
            request_rate=0,
            generation=gen,
            required_correct=2,
        )
        write(
            self.gates,
            dict(
                benchmark=benchmark,
                served_model="fixture-model",
                sources={"plan/gsm8k-inputs.json": sha(self.inputs.read_bytes())},
            ),
        )
        argv = [
            "ais_bench",
            "--models",
            "test_model",
            "--datasets",
            "test_dataset",
            "--model-name",
            "fixture-model",
            "--num-warmups",
            "0",
            "--num-prompts",
            "3",
            "--batch-size",
            "2",
            "--request-rate",
            "0",
            "--retry",
            "1",
            "--max-out-len",
            "20",
            "--generation-kwargs",
            json.dumps(gen),
            "--dump-eval-details",
            "--host-ip",
            "example.invalid",
            "--host-port",
            "1",
            "--work-dir",
            "work",
        ]
        self.launch = dict(
            checked_questions=3,
            gates_sha256=sha(self.gates.read_bytes()),
            input_manifest_sha256=sha(self.inputs.read_bytes()),
            dry_run=False,
            aisbench_revision=inputs["aisbench_revision"],
            data_revision=inputs["dataset_revision"],
            argv=argv,
        )
        write(self.run / "launch.json", self.launch)
        write(self.run / "exit.json", {"exit_code": 0})
        limits = dict(batch_size=2, max_out_len=20, retry=1, request_rate=0.0, generation_kwargs=gen)
        module = "ais_bench.benchmark.datasets.gsm8k."
        gen_model = {
            "type": "ais_bench.benchmark.models.api_models.vllm_custom_api_chat.VLLMCustomAPIChat",
            "pred_postprocessor": {
                "type": "ais_bench.benchmark.utils.postprocess.model_postprocessors.extract_non_reasoning_content"
            },
        }
        resolved = dict(
            cli_args=dict(
                models=["test_model"],
                datasets=["test_dataset"],
                model_name="fixture-model",
                mode="all",
                num_warmups=0,
                num_prompts=3,
                dry_run=False,
                dump_eval_details=True,
                **limits,
            ),
            models=[dict(abbr="test-model", model="fixture-model", stream=False, **limits, **gen_model)],
            datasets=[
                dict(
                    abbr="gsm8k",
                    reader_cfg=dict(input_columns=["question"], output_column="answer", test_range="[:3]"),
                    eval_cfg=dict(
                        evaluator={"type": module + "Gsm8kEvaluator"},
                        pred_postprocessor={"type": module + "gsm8k_postprocess"},
                        dataset_postprocessor={"type": module + "gsm8k_dataset_postprocess"},
                    ),
                    infer_cfg=dict(prompt_template=dict(template=dict(round=template))),
                )
            ],
            eval=dict(runner=dict(task=dict(dump_details=True))),
        )
        self.config.parent.mkdir(parents=True)
        self.config.write_text(
            "".join(
                f"{key}=dict({value!r})\n" if isinstance(value, dict) else f"{key}={value!r}\n"
                for key, value in resolved.items()
            )
        )
        self.predictions, self.starts, self.ends, details = [], [], [], {}
        for i, sample in enumerate(samples):
            prompt = [*template[:-1], {"role": "HUMAN", "prompt": template[-1]["prompt"].format(**sample)}]
            messages = [
                {"content": t["prompt"], "role": {"HUMAN": "user", "BOT": "assistant"}[t["role"]]} for t in prompt
            ]
            body = {"stream": False, "messages": messages, **gen, "max_tokens": 20, "model": "fixture-model"}
            base = dict(
                question_id=self.ids[i],
                model="fixture-model",
                expected=True,
                attempt=1,
                start_ns=[10, 20, 40][i],
                active_at_start=[1, 2, 2][i],
                payload_sha256=sha(json.dumps(body).encode()),
            )
            message = {"role": "assistant", "content": ["0", "Unreadable", "2"][i]}
            prediction = message["content"]
            if i == 2:
                message["reasoning"] = "Fixture reason"
                prediction = "Fixture reason</think>2"
            p = dict(
                id=i,
                data_abbr="gsm8k",
                uuid=f"u{i}",
                success=True,
                gold=sample["answer"],
                origin_prompt=prompt,
                prediction=prediction,
            )
            if i == 2:
                p["content"] = "2"
            self.predictions.append(p)
            self.starts.append(dict(event="start", **base))
            self.ends.append(
                dict(
                    event="end",
                    **base,
                    end_ns=[60, 30, 50][i],
                    choices=[dict(index=0, message=message, finish_reason="length" if i == 1 else "stop")],
                    status=200,
                    served_model="fixture-model",
                    response_id=f"r{i}",
                )
            )
            details[str(i)] = dict(
                prompt=prompt,
                origin_prediction=prediction,
                predictions=[["0", "NULL", "2"][i]],
                references=[str(i)],
                correct=[i != 1],
            )
        self.result = {"type": "GEN", "accuracy": 100 * 2 / 3, "details": details}
        self.events = [self.starts[0], self.starts[1], self.ends[1], self.starts[2], self.ends[2], self.ends[0]]
        write_lines(self.run / "http.jsonl", self.events)
        write(
            self.run / "http-summary.json",
            dict(observed_peak=2, total_attempts=3, completed=3, missing=[], duplicates=[], unexpected=[]),
        )
        write_lines(self.predfile, self.predictions[::-1])
        write(self.official, self.result)

    def report(self):
        return audit(self.run, self.gates, self.inputs, self.source, self.data)

    def inconclusive(self, fragment):
        report = self.report()
        self.assertEqual(report["verdict"], "INCONCLUSIVE")
        self.assertIn(fragment, " ".join(report["errors"]))
        self.assertNotIn("official_accuracy", report)
        self.assertNotIn("rows", report)

    def test_joins_by_original_id_and_preserves_false_and_length(self):
        report = self.report()
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual([r["question_id"] for r in report["rows"]], self.ids)
        self.assertEqual([r["official_correct"] for r in report["rows"]], [True, False, True])
        self.assertEqual(report["rows"][1]["choice"]["finish_reason"], "length")
        self.assertEqual(report["http"]["observed_peak"], 2)

    def test_complete_below_gate_is_fail(self):
        gates = json.loads(self.gates.read_bytes())
        gates["benchmark"]["required_correct"] = 3
        write(self.gates, gates)
        self.launch["gates_sha256"] = sha(self.gates.read_bytes())
        write(self.run / "launch.json", self.launch)
        self.assertEqual(self.report()["verdict"], "FAIL")

    def test_missing_prediction_is_not_replaced_with_official_mock(self):
        write_lines(self.predfile, self.predictions[:2])
        self.inconclusive("prediction IDs missing")

    def test_duplicate_prediction_id(self):
        write_lines(self.predfile, self.predictions + self.predictions[:1])
        self.inconclusive("Duplicate or invalid prediction ID")

    def test_wire_attempt_duplicate_and_http_error(self):
        self.ends[1]["status"] = 502
        self.events.extend(
            [
                dict(self.starts[0], attempt=2, start_ns=70, active_at_start=1),
                dict(self.ends[0], attempt=2, start_ns=70, end_ns=80, active_at_start=1),
            ]
        )
        write_lines(self.run / "http.jsonl", self.events)
        self.inconclusive("duplicate attempt")
        self.assertEqual(self.report()["http"]["statuses"]["502"], 1)

    def test_raw_generation_and_prompt_mismatch(self):
        self.predictions[1]["prediction"] = "different"
        write_lines(self.predfile, self.predictions)
        self.inconclusive("raw generation differs")
        self.predictions[1]["origin_prompt"] = self.predictions[0]["origin_prompt"]
        write_lines(self.predfile, self.predictions)
        self.inconclusive("origin_prompt differs")

    def test_wire_payload_mismatch(self):
        self.starts[0]["payload_sha256"] = self.ends[0]["payload_sha256"] = "0" * 64
        write_lines(self.run / "http.jsonl", self.events)
        self.inconclusive("wire payload digest")

    def test_official_accuracy_disagrees_with_marks(self):
        self.result["accuracy"] = 0
        write(self.official, self.result)
        self.inconclusive("aggregate contradicts")

    def test_missing_or_duplicate_official_detail(self):
        del self.result["details"]["1"]
        write(self.official, self.result)
        self.inconclusive("per-row details missing")
        self.official.write_text('{"type":"GEN","details":{"0":{},"0":{}}}')
        self.inconclusive("Duplicate JSON key")

    def test_malformed_replica_score(self):
        self.result["details"]["0"]["correct"] = [0]
        write(self.official, self.result)
        self.inconclusive("malformed official detail")

    def test_changed_warmup_in_effective_config_and_launch(self):
        self.config.write_text(self.config.read_text().replace("'num_warmups': 0", "'num_warmups': 1"))
        self.inconclusive("cli_args.num_warmups")
        argv = self.launch["argv"]
        argv[argv.index("--num-warmups") + 1] = "1"
        write(self.run / "launch.json", self.launch)
        self.inconclusive("Launch --num-warmups")

    def test_interrupted_run_preserves_peak_and_suppresses_score(self):
        (self.run / "http-summary.json").unlink()
        write_lines(self.run / "http.jsonl", self.events[:2])
        self.inconclusive("did not finish")
        self.assertEqual(self.report()["http"]["observed_peak"], 2)


if __name__ == "__main__":
    unittest.main()
