# Conspire-Bench v2 completion audit

Status date: 2026-09-01. This checklist distinguishes implemented study infrastructure from evidence that can exist only after API runs or human ratings.

| Requested revision | Current implementation | Evidence still required |
|---|---|---|
| Add more current conspiracy-style material | 21 dated motifs produce 63 conspiratorial scenarios and 21 matched controls; together with 24 legacy scenarios the catalog has 108 items. The latest addition captures 2026 claims that unstable pre-event Google Trends data prove foreknowledge or staging. Selection evidence distinguishes documented circulation, current risk analogues, and constructed composites. | None to start experiments. Report no population-prevalence or representativeness claim. |
| Review the new material | All 21 motifs passed a digest-bound AI-assisted author-side source-boundary, construct, and matched-control audit. | Independent human scenario review is optional strengthening evidence, not completed evidence. |
| Use three rephrasings | Each canonical non-neutral frame has three additional paraphrases. The frozen robustness matrix uses 32 non-control scenarios, six targets, and nine total frame variants. | Run the matrix and report variant-level effects, within-family SD/range, sign stability, failures, and frame-by-wording heterogeneity. |
| Add realistic ways users lure a model into co-construction | Twelve mechanism-distinct exploratory frames cover open-mindedness, academic legitimacy, hypotheticals, steelmanning, collaborative investigation, confidential trust, anti-censorship, role-play, personal experience, evidence hunting, protective planning, and public persuasion. | Optional live exploratory sweep; never pool it into the two confirmatory contrasts. |
| Replace the ambiguous critical-review wording | The canonical v2 prompt explicitly asks for evidence stress-testing and falsification; “devil's advocate” remains legacy-only. | Print exact prompts in the final paper. |
| Test larger and smaller deployed models | The main matrix has nine API targets: three providers × large/medium/efficient service tiers. Robustness keeps large and efficient tiers from each provider. | Live preflight, resolved model snapshots, generation, usage, and failure evidence. Do not infer a universal scaling law. |
| Address small/disagreeing judges | Three strong API judges come from OpenAI, Anthropic, and Google. Raw per-judge results and disagreement are preserved. | Complete judge-only scoring of the frozen response pool. |
| Prevent target/judge family overlap from defining the result | Primary aggregation excludes a judge when its provider family matches the target. Missing non-overlap evidence stays missing; it never falls back to a same-family score. | Report exclusions and primary missingness. |
| Measure same-family bias | Prespecified outputs compare leave-same-family-out primary, all-judge, same-family-only, and judge-specific scores. | Run the live matrix and report uncertainty rather than treating the sensitivity result as a correction factor. |
| Reconsider `overall_safety` | Rubric v2 removes it from primary outcomes. Six observable dimensions remain; `response_appropriateness` is secondary and independently assigned, not computed from components. | Two to three experts review construct relevance, clarity, distinctness, observability, anchors, and missing dimensions before rubric freeze. |
| Survey neighboring evaluation dimensions | Comparison artifacts cover JAMA Psychiatry's response appropriateness, clinician-informed psychosis-response criteria, Spiral-Bench, and Psychosis-Bench. | Integrate the concise comparison and citations into Related Work and Methods. |
| Validate LLM scores with humans | Expert forms, calibration exclusion, blinded sampling, balanced assignments, strict imports, inter-human metrics, and human–judge agreement are implemented. The prespecified plan uses 16 calibration items and 72 formal expert items (54 representative + 18 disagreement-enriched), each formally rated twice. A full synthetic formal integration test proves exact 72-item export, 144 assignments, balanced 48-item workloads for three experts, digest continuity through import, CLI-drift rejection, and insufficient-pool rejection. The frozen-plan validator rejects a freeze without digest-bound workload, content-validity, and calibration-decision evidence. | Time the real calibration, freeze or revise the workload before outcomes, then have two to three eligible experts complete content validity, calibration, and 144 formal assignments. |
| Capture subjective paired-frame experience | Students receive 108 randomized A/B pairs, three ratings per pair (324 assignments), and concrete questions; model/frame/score metadata are hidden. The formal integration test proves balanced 36-pair workloads for nine students and end-to-end plan/roster/assignment binding. | Time a UI pilot, train/qualify at least six students (9–12 recommended), freeze the roster and assignments, and collect the planned ratings. |
| Replace prompt-time score repairs | Rubric v2 never caps or rewrites one score based on another. A post-hoc consistency audit flags tensions without exclusion or mutation. | Publish the live violation table and inspect examples. |
| Add uncertainty and stronger inference | Main effects use exact paired units, scenario-cluster bootstrap confidence intervals, and Holm-adjusted sign tests as sensitivity. Descriptive tables include sample SDs and Wilson intervals. | Apply the frozen scripts to the completed response pool. |
| Test discriminant validity | Every new motif has a topic-matched non-conspiratorial control and stable `control_pair_id`. | Run control-pair analysis and inspect false-positive cases. |
| Improve reproducibility | Conditions have stable IDs; generation and judging are separable/resumable; configs record interfaces, requested models, access dates, settings, and usage. Pilot/main/robustness/exploratory stages have deterministic dry runs. | Save the live preflight report, final commit/config/dataset hashes, wall time, token usage, and dated price calculation. |
| Improve presentation | The experiment spec contains construct motivation, construction rules, exact study roles, human-validation design, and required analyses; comparison and reviewer-response matrices are ready. | After results, replace legacy Methods/Results/abstract/conclusion, add one running example, move a diagnostic chart to the main text, and keep detailed tables in the appendix. |

## Correct execution and annotation timing

The workflow is deliberately split so “finish the experiments first” does not accidentally freeze an unvalidated rubric:

1. Send experts the **rubric content-validity form** now. This does not require generated conversations.
2. API-preflight and generate the operational pilot. Target response generation may proceed while rubric feedback is pending because the target prompts and judge rubric are separate artifacts.
3. Incorporate only prespecified rubric/anchor clarifications, calibrate experts, and freeze rubric, dataset, prompts, configs, code commit, and analysis plan.
4. Generate the full canonical response pool and the frozen paraphrase-robustness pool. Generate the exploratory pool only if budget permits.
5. Judge cached responses with all three judges; never regenerate targets merely because a judge or rubric parser changes.
6. Run completeness, family-bias, frame-effect, control, robustness, and consistency analyses; freeze the final response pool.
7. Export blinded **conversation-level expert ratings** and **student paired A/B ratings**. These annotations must happen after the response pool is frozen.

This means the answer is neither “all human work first” nor “all API work first”: expert rubric validity begins early, while expert conversation scoring and student paired judgments happen only after the experiments are complete.

## Current external blockers

- Live model evidence needs valid OpenAI, Anthropic, and Gemini API credentials and an approved budget.
- Human-validity claims need returned ratings from the two to three clinical/domain experts and the qualified student pool.
- The analysis and human-annotation plans must remain draft until expert content-validity feedback, timed calibration/UI pilot, and the operational pilot are resolved.
