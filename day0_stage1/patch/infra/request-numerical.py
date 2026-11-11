"""Send the frozen non-smoke prompt to the service with an optional worker trace."""
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base-url", required=True)
p.add_argument("--output", required=True, type=Path)
args = p.parse_args()
prompts = json.loads(Path("/stage1/validator/prompts.json").read_text())
case = next(c for c in prompts["cases"] if c["id"] == "numerical")
body = dict(model=prompts["served_model"], prompt=case["prompt_ids"],
            temperature=0, max_tokens=prompts["generation"]["oracle_max_tokens"], stream=False)
with args.output.open("x") as log:
    req = Request(args.base_url.rstrip("/") + "/completions", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=prompts["smoke_timeout_s"]) as response:
        data = json.load(response)
    json.dump(dict(request=body, response=data), log)
    if (data["model"] != body["model"] or not data["id"] or
        data["usage"]["prompt_tokens"] != len(case["prompt_ids"]) or
        data["choices"][0]["finish_reason"] not in {"length", "stop"}):
        raise SystemExit("Unexpected generation response")
