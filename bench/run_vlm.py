"""Ask a vision-language model the SAED-Bench questions.

Anthropic:
    export ANTHROPIC_API_KEY=...
    python -m bench.run_vlm --provider anthropic --model claude-sonnet-5-5

Any OpenAI-compatible endpoint (OpenAI, OpenRouter, Google's OpenAI-compatible API, a local vLLM/Ollama server):
    export OPENAI_API_KEY=...
    python -m bench.run_vlm --provider openai --model <model-name> [--base-url https://openrouter.ai/api/v1]

Results are appended to bench/data/v0/results_<model>.jsonl and resumed if the run is interrupted.
Use --limit 8 for a cheap smoke test and --dry-run to print the first prompt without calling any API.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from saedscope.crystal import builtin_library

from .common import SYSTEM, extract_answer, grade, load_items

LIB = {p.name: p for p in builtin_library()}


def ask_anthropic(client, model, prompt, png_b64, max_tokens):
    msg = client.messages.create(
        model=model, max_tokens=max_tokens, system=SYSTEM,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64}},
            {"type": "text", "text": prompt}]}])
    return "".join(getattr(b, "text", "") for b in msg.content)


def ask_openai(client, model, prompt, png_b64, max_tokens):
    r = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": [
                      {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                      {"type": "text", "text": prompt}]}])
    return r.choices[0].message.content or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["anthropic", "openai"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--data", default="bench/data/v0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tasks", default="", help="comma-separated subset of tasks")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = Path(args.data)
    items = load_items(data / "items.jsonl")
    if args.tasks:
        keep = set(args.tasks.split(","))
        items = [i for i in items if i["task"] in keep]
    if args.limit:
        # balanced across tasks
        by = {}
        for it in items:
            by.setdefault(it["task"], []).append(it)
        items = [it for t in by.values() for it in t[: max(1, args.limit // len(by))]]
    if args.dry_run:
        print(SYSTEM, "\n---\n", items[0]["prompt"], "\n(image:", items[0]["image"], ")")
        return

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    out = data / f"results_{safe}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out) if l.strip()}
    todo = [i for i in items if i["id"] not in done]
    print(f"{len(todo)} items to run ({len(done)} already done) → {out}")

    if args.provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        ask = ask_anthropic
    else:
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url or None,
                        api_key=os.environ.get("OPENAI_API_KEY") or ("not-needed" if args.base_url else None))
        ask = ask_openai

    def run(it):
        png = base64.b64encode((data / it["image"]).read_bytes()).decode()
        for attempt in range(4):
            try:
                t0 = time.time()
                text = ask(client, args.model, it["prompt"], png, args.max_tokens)
                ans = extract_answer(text)
                return {"id": it["id"], "task": it["task"], "model": args.model, "answer": ans, "truth": it["answer"],
                        "correct": grade(it, ans, phase_lookup=LIB.get), "seconds": round(time.time() - t0, 1), "raw": text[-1500:]}
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                time.sleep(2 * (attempt + 1))
        return {"id": it["id"], "task": it["task"], "model": args.model, "answer": "", "truth": it["answer"], "correct": False, "error": err}

    with open(out, "a") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, it) for it in todo]
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{k}/{len(todo)}] {r['id']:16s} {str(r['answer'])[:24]:24s} truth={str(r['truth']):22s} {'✓' if r['correct'] else '✗'}")


if __name__ == "__main__":
    main()
