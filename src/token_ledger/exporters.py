# llm_tracker/exporters.py
"""
Handles all output: console printing and file writing.
Completely decoupled from all other modules.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from .trace import Trace


def print_summary(summary: dict) -> None:
    w = 54
    print("\n" + "=" * w)
    print("  LLM COST SUMMARY")
    print("=" * w)
    print(f"  Project      : {summary['project_name']}")
    print(f"  Total Calls  : {summary['total_calls']}")
    print(f"  Input Tokens : {summary['total_input_tokens']:,}")
    print(f"  Output Tokens: {summary['total_output_tokens']:,}")
    print(f"  Total Tokens : {summary['total_tokens']:,}")
    print("-" * w)
    print(f"  Total Cost      : ${summary['total_cost']:.6f}")
    print(f"  Embedding Cost  : ${summary['embedding_cost']:.6f}")
    print(f"  Retry Cost      : ${summary['retry_cost']:.6f}")
    if summary["budget"]:
        status = "OK" if (summary["budget_remaining"] or 0) >= 0 else "EXCEEDED"
        print(f"  Budget          : ${summary['budget']:.4f} [{status}]")
        print(f"  Remaining       : ${summary.get('budget_remaining', 0):.6f}")
    print("-" * w)
    print("  Cost by Model:")
    for model, cost in sorted(summary["cost_by_model"].items(), key=lambda x: -x[1]):
        print(f"    {model:<40} ${cost:.6f}")
    print("=" * w)


def write_file(
    project_name: str,
    summary:      dict,
    traces:       List[Trace],
    output_dir:   str = ".",
) -> None:
    """Write full cost report to {project_name}_token_cost.txt"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    filepath = Path(output_dir) / f"{project_name}_token_cost.txt"

    with open(filepath, "w", encoding="utf-8") as f:
        # ── Header ────────────────────────────────────────────
        f.write("=" * 60 + "\n")
        f.write(f"LLM TOKEN COST REPORT — {project_name}\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("=" * 60 + "\n\n")

        # ── Summary ───────────────────────────────────────────
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Calls    : {summary['total_calls']}\n")
        f.write(f"Total Tokens   : {summary['total_tokens']:,}\n")
        f.write(f"  Input        : {summary['total_input_tokens']:,}\n")
        f.write(f"  Output       : {summary['total_output_tokens']:,}\n")
        f.write(f"Total Cost     : ${summary['total_cost']:.6f}\n")
        f.write(f"Embedding Cost : ${summary['embedding_cost']:.6f}\n")
        f.write(f"Retry Cost     : ${summary['retry_cost']:.6f}\n")
        if summary["budget"]:
            f.write(f"Budget         : ${summary['budget']:.4f}\n")
            f.write(f"Remaining      : ${summary.get('budget_remaining', 0):.6f}\n")
        f.write("\nCost by Model:\n")
        for model, cost in sorted(summary["cost_by_model"].items(), key=lambda x: -x[1]):
            f.write(f"  {model:<42} ${cost:.6f}\n")

        # ── Call tree ─────────────────────────────────────────
        f.write("\n\nCALL LOG (chronological)\n")
        f.write("-" * 40 + "\n")
        for trace in traces:
            indent = "  " * trace.depth
            est    = " [estimated]" if trace.usage.is_estimated else ""
            retry  = f" retries={trace.retry_count}" if trace.retry_count else ""
            f.write(
                f"{indent}[{trace.call_type[0].upper()}] "
                f"{trace.name or trace.model} | "
                f"in={trace.usage.input_tokens} "
                f"out={trace.usage.output_tokens} | "
                f"${trace.cost.total_cost:.6f}"
                f"{est}{retry}"
                f" @ {trace.timestamp}\n"
            )

        # ── Raw JSON ──────────────────────────────────────────
        f.write("\n\nRAW DATA (JSON)\n")
        f.write("-" * 40 + "\n")
        json_data = {
            "summary": summary,
            "traces":  [t.to_dict() for t in traces],
        }
        f.write(json.dumps(json_data, indent=2))

    print(f"\n[llm_tracker] Report saved → {filepath}")