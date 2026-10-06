"""
@file export_rule_review.py
@description Export the advisory rulebook as a review sheet for an agronomist or KVK scientist.
@module scripts

Every rule is listed with its condition in plain English, its advice in English and Hindi,
its confidence tag, and the document it should be verified against. The reviewer marks each
rule; a verified rule is promoted by changing `confidence: ASSUMPTION` to `SOURCED` and
replacing the citation with the document and page.

RUN
---
    python scripts/export_rule_review.py
Writes:
    docs/rulebook_review.md    readable checklist, grouped by crop
    docs/rulebook_review.csv   the same, for a spreadsheet
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def main() -> None:
    from services.advisory_engine.evaluator import describe_condition
    from services.advisory_engine.phenology import stage_label
    from services.advisory_engine.rulebook import load_rulebook

    rules = load_rulebook(refresh=True)
    by_crop: dict[str, list] = defaultdict(list)
    for r in rules:
        by_crop[r.crop].append(r)
    sourced = sum(1 for r in rules if r.confidence_tag == "SOURCED")

    out_dir = ROOT_DIR / "docs"
    out_dir.mkdir(exist_ok=True)

    md: list[str] = [
        "# Advisory rulebook — agronomist review sheet",
        "",
        f"> Generated {date.today().isoformat()} from `services/advisory_engine/rulebooks/` "
        f"by `scripts/export_rule_review.py`. Do not edit by hand; regenerate after changing rules.",
        "",
        f"**{len(rules)} rules across {len(by_crop)} crops — {sourced} SOURCED, "
        f"{len(rules) - sourced} ASSUMPTION.**",
        "",
        "## How to review",
        "",
        "For each rule, check the **threshold** and the **advice** against the document named in",
        "*Verify against*. Then, in the crop's YAML file:",
        "",
        "- **Confirmed:** set `confidence: SOURCED` and replace `citation` with the document and page.",
        "- **Wrong threshold:** correct `value` in the `when` block and cite the source.",
        "- **Wrong advice:** correct the text in `rulebooks/messages.yaml`.",
        "- **Not applicable:** delete the rule.",
        "",
        "Then run `python -m services.advisory_engine` to validate, and re-run this script.",
        "",
    ]

    rows = []
    for crop in sorted(by_crop):
        md += [f"## {crop.capitalize()}", ""]
        for r in by_crop[crop]:
            stages = ", ".join(stage_label(s)[0] for s in r.stages)
            condition = describe_condition(r.when)
            md += [
                f"### `{r.rule_id}` — {r.risk_level}",
                "",
                f"- **Stage:** {stages}",
                f"- **Fires when:** {condition}",
                f"- **Advice (en):** {r.recommended_action}",
                f"- **Advice (hi):** {r.vernacular_summary.get('hi', '—')}",
                f"- **Confidence:** {r.confidence_tag}",
                f"- **Citation:** {r.citation}",
                f"- **Verify against:** {r.verify_against or '—'}",
            ]
            if r.note:
                md.append(f"- **Note:** {r.note}")
            md += ["- **Reviewer:** ☐ confirmed  ☐ threshold changed  ☐ advice changed  ☐ remove", ""]
            rows.append({
                "crop": crop, "rule_id": r.rule_id, "risk": r.risk_level, "stages": stages,
                "condition": condition, "advice_en": r.recommended_action,
                "advice_hi": r.vernacular_summary.get("hi", ""), "confidence": r.confidence_tag,
                "citation": r.citation, "verify_against": r.verify_against, "note": r.note,
                "reviewer_decision": "", "reviewer_source": "",
            })

    (out_dir / "rulebook_review.md").write_text("\n".join(md), encoding="utf-8")
    with open(out_dir / "rulebook_review.csv", "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote docs/rulebook_review.md and docs/rulebook_review.csv ({len(rules)} rules)")


if __name__ == "__main__":
    main()
