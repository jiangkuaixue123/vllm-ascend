"""Check the advertised alias and tokenize unchanged chat requests through the real service."""

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from stage1_trace import HERE, dump_json


def fetch(url, timeout, payload=None):
    request = Request(url) if payload is None else Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}
    )
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}")
        return json.load(response)


def check(api_base, timeout):
    config = json.loads((HERE / "prompts.json").read_bytes())
    model = config["served_model"]
    endpoint = urlsplit(api_base.rstrip("/"))
    if endpoint.path.rsplit("/", 1)[-1] != "v1" or endpoint.query or endpoint.fragment:
        raise ValueError("expected an API base ending in /v1")
    root = urlunsplit(endpoint._replace(path=endpoint.path.rsplit("/", 1)[0]))
    models = fetch(api_base.rstrip("/") + "/models", timeout)
    aliases = [item["id"] for item in models["data"]]
    rows = []
    for case in config["cases"]:
        # This must exercise service defaults, exactly like smoke_chat.py. Adding
        # chat_template_kwargs here would mask an incorrect serve configuration.
        payload = {"model":model, "messages":case["messages"]}
        data = fetch(root + "/tokenize", timeout, payload)
        ids = data["tokens"]
        rows.append({"case":case["id"], "ids":ids, "matches_frozen_ids":ids==case["prompt_ids"] and data["count"]==len(ids)})
    return dict(advertised_aliases=aliases, model=model, tokens=rows,
                passed=model in aliases and all(row["matches_frozen_ids"] for row in rows))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url",required=True)
    p.add_argument("--timeout",type=float,default=600)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    try:
        result=check(args.base_url,args.timeout)
    except (ValueError, KeyError, TypeError, OSError) as ex:
        result={"passed":False, "error":type(ex).__name__}
        if isinstance(ex, HTTPError):
            result["http_status"]=ex.code
    dump_json(args.output,result)
    raise SystemExit(0 if result["passed"] else 1)
