"""Regenerate the generated parts of the `docs/` static site (GitHub Pages,
served from `/docs` on `main`) from `results/`.

`docs/index.html`, `docs/run.html`, `docs/app.js`, `docs/style.css` are
hand-written and never touched by this script. Only two things are
(re)generated here:

- `docs/data.json`: a copy of `results/recap.json` (run `scripts/build_recap.py`
  first if it's missing or stale).
- `docs/results/`: a full mirror of `results/` (every `trace.jsonl`,
  `calls.jsonl`/legacy call logs, and `run.json`), so the site is entirely
  self-contained under `docs/` -- the simplest GitHub Pages setup ("serve
  `/docs` from `main`") only serves files inside that folder, so `results/`
  at the repo root would otherwise be unreachable from the published site.

Run via `make site`, or by CI after `make recap` on every push to `main`
that touches `results/**`.
"""

import shutil
from pathlib import Path

RESULTS_ROOT = Path("results")
DOCS_ROOT = Path("docs")
EXCLUDED_TOP_LEVEL_FILES = {"RECAP.md", "recap.json"}


def main() -> None:
    recap_json = RESULTS_ROOT / "recap.json"
    if not recap_json.exists():
        raise SystemExit(f"{recap_json} not found -- run `make recap` (scripts/build_recap.py) first.")
    shutil.copyfile(recap_json, DOCS_ROOT / "data.json")

    mirror_root = DOCS_ROOT / "results"
    if mirror_root.exists():
        shutil.rmtree(mirror_root)
    mirror_root.mkdir(parents=True)

    copied = 0
    for path in RESULTS_ROOT.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(RESULTS_ROOT)
        if len(relative.parts) == 1 and relative.name in EXCLUDED_TOP_LEVEL_FILES:
            continue
        dest = mirror_root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        copied += 1

    print(f"Wrote {DOCS_ROOT}/data.json and mirrored {copied} file(s) into {mirror_root}")


if __name__ == "__main__":
    main()
