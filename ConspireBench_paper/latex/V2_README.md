# Manuscript status: do not resubmit `acl_latex.tex` as v2

`acl_latex.tex` and its compiled PDF reproduce the May 2026 submission. They intentionally contain the legacy 24-scenario, six-local-target, two-local-judge experiment and rubric v1 `overall_safety` results.

The repository’s v2 dataset, rubric, prompts, model matrix, and annotation workflow are newer than this manuscript. They do not yet have live API results or completed human validation. Therefore:

- do not edit legacy result numbers into v2 terminology;
- describe the 108-scenario catalog as AI-assisted author-reviewed, not independently human-reviewed, unless a separate reviewer later completes the optional review package;
- do not claim strong-judge, family-excluded, paraphrase-robust, or human-aligned results until the corresponding frozen analyses exist;
- do not use `overall_safety` as a v2 endpoint;
- do not submit the current PDF as the revised paper.

Use `docs/reviewer_response_matrix_v2.md` as the replacement checklist. After the v2 response pool and annotations are frozen, create the revised manuscript from the following evidence:

1. `configs/analysis_plan_v2.json` plus its completed freeze record;
2. main and robustness result manifests with requested/resolved model IDs and access dates;
3. `analysis/frame_effect_stats.py`, `analysis/control_pair_validity.py`, `analysis/paraphrase_robustness.py`, and `analysis/judge_family_sensitivity.py` outputs;
4. human agreement output from `analysis/import_annotations.py`;
5. the digest-bound AI-assisted author QA ledger, with independent human scenario review described only if separately completed;
6. the expert content-validity report and calibration decision used to freeze rubric v2.

`v2_methods_draft.tex` already contains result-free Methods text that matches the current implementation. `docs/running_example_v2.md` fixes the example motif, source boundary, matched control, exact canonical frames, and result fields; fill its response IDs, excerpts, and scores only from the frozen pool.

The old experiment can remain as a labelled legacy baseline or appendix, but rubric-v1 and rubric-v2 scores must not be pooled.
