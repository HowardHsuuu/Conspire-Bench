# Expert rubric content-validity packet

This packet can be sent before any model experiment is complete. It asks experts to evaluate the Conspire-Bench v2 rubric itself, not model responses and not any user's mental state.

Send each expert:

- one private copy of either `expert_rubric_feedback.json` or `expert_rubric_feedback.csv`;
- `docs/rubric.md` as the full rubric manual;
- the study's consent, compensation, data-retention, withdrawal, and contact information.

Each expert should use a pseudonymous `expert_id`, briefly describe relevant expertise without identifying information, rate all seven listed dimensions, choose `retain`, `revise`, or `remove`, and complete the `__global__` row or `global_feedback` object. The scale definitions and anchors are embedded in both formats.

Do not combine experts in one returned file. Suggested filenames are `rubric_expert_01.json`, `rubric_expert_02.json`, and `rubric_expert_03.json`.

After receiving the forms, run:

```bash
python3 analysis/import_rubric_feedback.py \
  --input returned/rubric_expert_01.json \
  --input returned/rubric_expert_02.json \
  --out-dir annotation/rubric_validity_analysis
```

Review every requested revision and document the decision. If definitions or anchors change, increment the rubric version and rerun calibration before final automated judging. The generated CVI summaries are descriptive for a panel of two or three experts; retain the raw counts and qualitative comments.
