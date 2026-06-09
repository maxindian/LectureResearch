#!/usr/bin/env python3
"""Run CO2RR pathway extraction benchmark with DeepSeek-V4 API."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"


def load_dataset(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_json_object(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def get_nested(data: dict, field: str) -> Any:
    cur: Any = data
    for part in field.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def set_overlap(pred: Any, gt_list: list) -> float:
    if not isinstance(pred, list) or not gt_list:
        return 0.0
    pred_set = {str(x).lower() for x in pred}
    gt_set = {str(x).lower() for x in gt_list}
    if not gt_set:
        return 1.0
    return len(pred_set & gt_set) / len(gt_set)


def run_check(pred: dict, check: dict) -> dict:
    cid = check["id"]
    ctype = check["type"]
    field = check.get("field", "")
    passed = False
    detail = ""

    if ctype == "exact":
        expected = check["expected"]
        val = get_nested(pred, field) if field else pred
        passed = val == expected
        detail = f"expected={expected!r}, got={val!r}"

    elif ctype == "set_overlap":
        val = get_nested(pred, field)
        recall = set_overlap(val, check["ground_truth"])
        passed = recall >= check.get("min_recall", 0.5)
        detail = f"recall={recall:.2f}"

    elif ctype == "numeric_tolerance":
        expected = check["expected"]
        tol = check.get("tolerance", 1)
        val = get_nested(pred, field)
        if val is None and field == "reaction_pathway.step_count":
            steps = get_nested(pred, "reaction_pathway.steps")
            val = len(steps) if isinstance(steps, list) else None
        if isinstance(val, (int, float)):
            passed = abs(val - expected) <= tol
            detail = f"expected≈{expected}±{tol}, got={val}"
        else:
            detail = f"non-numeric: {val!r}"

    elif ctype == "substring":
        expected = check["expected_substring"]
        val = get_nested(pred, field)
        if isinstance(val, list):
            blob = json.dumps(val, ensure_ascii=False)
        else:
            blob = str(val or "")
        passed = expected in blob
        detail = f"substring {expected!r} in field"

    else:
        passed = False
        detail = f"unknown check type: {ctype}"

    return {"id": cid, "type": ctype, "passed": passed, "detail": detail}


def score_prediction(pred: dict | None, verification: dict) -> dict:
    if pred is None:
        return {
            "checks_passed": 0,
            "checks_total": len(verification.get("checks", [])),
            "pass_ratio": 0.0,
            "passed": False,
            "check_results": [],
        }
    results = [run_check(pred, c) for c in verification.get("checks", [])]
    passed_n = sum(1 for r in results if r["passed"])
    total = len(results) or 1
    ratio = passed_n / total
    min_ratio = verification.get("min_pass_ratio", 0.6)
    return {
        "checks_passed": passed_n,
        "checks_total": total,
        "pass_ratio": ratio,
        "passed": ratio >= min_ratio,
        "check_results": results,
    }


def call_deepseek(
    client: Any,
    model: str,
    prompt: str,
    *,
    thinking: bool,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict]:
    extra: dict = {}
    if thinking:
        extra["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        extra["extra_body"] = {"thinking": {"type": "disabled"}}

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是电催化 CO2RR 领域的结构化信息抽取助手。"
                    "只输出合法 JSON，不要附加解释。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        **extra,
    )
    content = response.choices[0].message.content or ""
    usage = {}
    if getattr(response, "usage", None):
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return content, usage


def run_eval(args: argparse.Namespace) -> dict:
    if OpenAI is None:
        raise RuntimeError("请安装 openai: pip install openai")

    api_key = args.api_key or os.getenv("DEEPSEEK_API_KEY")
    # sk-a2e9ca823b714d0e99a8d8c12c008613
    if not api_key:
        raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY 或使用 --api-key")

    dataset = load_dataset(args.eval_set)
    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)

    samples = dataset["samples"]
    if args.limit:
        samples = samples[: args.limit]
    if args.folder_ids:
        ids = {int(x) for x in args.folder_ids.split(",")}
        samples = [s for s in samples if s["folder_id"] in ids]

    results: list[dict] = []
    for sample in samples:
        sid = sample["id"]
        if sample["status"] != "ok":
            results.append(
                {
                    "id": sid,
                    "folder_id": sample["folder_id"],
                    "status": "skipped",
                    "reason": sample["status"],
                }
            )
            continue

        prompt = sample["input"]["prompt"]
        t0 = time.time()
        try:
            raw, usage = call_deepseek(
                client,
                args.model,
                prompt,
                thinking=args.thinking,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            pred = extract_json_object(raw)
            latency = time.time() - t0
            score = score_prediction(pred, sample["verification"])
            results.append(
                {
                    "id": sid,
                    "folder_id": sample["folder_id"],
                    "status": "completed",
                    "model": args.model,
                    "latency_sec": round(latency, 2),
                    "usage": usage,
                    "prediction": pred,
                    "raw_response_preview": raw[:500],
                    "score": score,
                }
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                {
                    "id": sid,
                    "folder_id": sample["folder_id"],
                    "status": "error",
                    "error": str(exc),
                }
            )
        if args.sleep > 0:
            time.sleep(args.sleep)

    completed = [r for r in results if r.get("status") == "completed"]
    passed = [r for r in completed if r.get("score", {}).get("passed")]
    summary = {
        "eval_set": str(args.eval_set),
        "model": args.model,
        "thinking": args.thinking,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(results),
        "completed": len(completed),
        "passed": len(passed),
        "pass_rate": len(passed) / len(completed) if completed else 0.0,
        "avg_pass_ratio": (
            sum(r["score"]["pass_ratio"] for r in completed) / len(completed)
            if completed
            else 0.0
        ),
        "results": results,
    }
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="CO2RR DeepSeek-V4 evaluation")
    parser.add_argument(
        "--eval-set",
        type=Path,
        default=root / "co2rr_eval.json",
        help="Path to co2rr_eval.json",
    )
    parser.add_argument("--output", type=Path, default=root / "co2rr_eval_results.json")
    parser.add_argument("--api-key", default=None, help="DeepSeek API key")
    parser.add_argument("--base-url", default=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL))
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--limit", type=int, default=None, help="Max samples to run")
    parser.add_argument("--folder-ids", default=None, help="Comma-separated folder ids, e.g. 1,2,3")
    parser.add_argument("--sleep", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--dry-run", action="store_true", help="Validate eval set only")
    args = parser.parse_args()

    if args.dry_run:
        data = load_dataset(args.eval_set)
        print(
            f"Eval set OK: {data['valid_sample_count']}/{data['sample_count']} valid samples"
        )
        return

    summary = run_eval(args)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Done. {summary['passed']}/{summary['completed']} passed "
        f"(pass_rate={summary['pass_rate']:.1%}, avg_check_ratio={summary['avg_pass_ratio']:.1%})"
    )
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
