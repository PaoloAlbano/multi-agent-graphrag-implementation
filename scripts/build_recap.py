"""Aggregate every `results/**/run.json` manifest into one summary.

Reads only `run.json` manifests (never re-derives accuracy by re-scanning
`trace.jsonl`, since the manifest is the self-describing source of truth
written by `cypherbench evaluate --run-manifest` or the legacy migration
script) and writes:

- `results/RECAP.md`: one Domain x {single, agentic, delta} table per
  model/config group, human-readable.
- `results/recap.json`: the same data, machine-readable, consumed by
  `scripts/build_site.py`.

Run via `make recap` after adding new `results/**` files (locally, or by CI
on every push to `main` that touches `results/**`).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

RESULTS_ROOT = Path("results")


def _leaf_group_key(run_json_path: Path) -> tuple[str, str]:
    """Return (model_slug, config_dir) for a `results/<model>/<config>/<domain>/<mode>/run.json`."""
    mode_dir = run_json_path.parent
    domain_dir = mode_dir.parent
    config_dir = domain_dir.parent
    model_dir = config_dir.parent
    return model_dir.name, config_dir.name


def _collect_groups() -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], dict] = {}

    for run_json_path in sorted(RESULTS_ROOT.rglob("run.json")):
        manifest = json.loads(run_json_path.read_text(encoding="utf-8"))
        model_slug, config_dir = _leaf_group_key(run_json_path)
        key = (model_slug, config_dir)
        group = groups.setdefault(
            key,
            {
                "model": manifest.get("model", model_slug),
                "model_slug": model_slug,
                "config": config_dir,
                "domains": {},
            },
        )

        leaf_dir = run_json_path.parent.relative_to(RESULTS_ROOT)
        for entry in manifest.get("results", []):
            domain = entry["domain"]
            mode = entry["mode"]
            domain_entry = group["domains"].setdefault(domain, {})
            domain_entry[mode] = {
                "total": entry["total"],
                "correct": entry["correct"],
                "accuracy": entry["accuracy"],
                "path": str(leaf_dir),
            }

    return groups


def _average(domains: dict[str, dict], mode: str) -> float | None:
    values = [d[mode]["accuracy"] for d in domains.values() if mode in d]
    return sum(values) / len(values) if values else None


def _build_recap_json(groups: dict[tuple[str, str], dict]) -> dict:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "groups": [
            {
                "model": group["model"],
                "model_slug": group["model_slug"],
                "config": group["config"],
                "domains": group["domains"],
                "average": {
                    "single": _average(group["domains"], "single"),
                    "agentic": _average(group["domains"], "agentic"),
                },
            }
            for group in groups.values()
        ],
    }


def _format_pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "-"


def _build_recap_md(recap: dict) -> str:
    lines = ["# Results recap", "", f"_Generated {recap['generated_at']}_", ""]
    for group in recap["groups"]:
        lines.append(f"## {group['model']} ({group['config']})")
        lines.append("")
        lines.append("| Domain | single | agentic | delta |")
        lines.append("|---|---|---|---|")
        for domain, modes in sorted(group["domains"].items()):
            single = modes.get("single", {}).get("accuracy")
            agentic = modes.get("agentic", {}).get("accuracy")
            delta = agentic - single if single is not None and agentic is not None else None
            delta_str = f"{delta:+.1%}" if delta is not None else "-"
            lines.append(f"| {domain} | {_format_pct(single)} | {_format_pct(agentic)} | {delta_str} |")

        avg_single = group["average"]["single"]
        avg_agentic = group["average"]["agentic"]
        avg_delta = avg_agentic - avg_single if avg_single is not None and avg_agentic is not None else None
        avg_delta_str = f"{avg_delta:+.1%}" if avg_delta is not None else "-"
        lines.append(
            f"| **Average** | {_format_pct(avg_single)} | {_format_pct(avg_agentic)} | {avg_delta_str} |"
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    groups = _collect_groups()
    recap = _build_recap_json(groups)

    (RESULTS_ROOT / "recap.json").write_text(json.dumps(recap, indent=2) + "\n", encoding="utf-8")
    (RESULTS_ROOT / "RECAP.md").write_text(_build_recap_md(recap), encoding="utf-8")

    print(f"Aggregated {len(groups)} model/config group(s) into {RESULTS_ROOT}/RECAP.md + recap.json")


if __name__ == "__main__":
    main()
