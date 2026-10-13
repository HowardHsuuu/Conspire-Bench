# Frozen V3 results and human-validation target

This document is the public interpretation record for the frozen automated V3
evidence. Exact file identities and claim restrictions are machine-readable in
`configs/study_freeze_v1.json`. The automated evidence is complete; human
validity evidence is not.

## Study identity

Conspire-Bench is a matched diagnostic benchmark of **interaction-frame
sensitivity**. It asks whether the same conspiracy narrative and interaction
trajectory elicit a different assistant epistemic posture when the user first
establishes a different conversational role. It does not estimate how common
those roles are, determine whether a conspiracy allegation is true, measure
belief change in users, diagnose users, rank assistants on one safety scale, or
certify deployment safety.

The frozen canonical matrix contains 51 circulation-grounded motifs, three
matched interaction structures, five frame families, and 17 target models:
13,005 conversations. Qwen 2.5 32B, Llama 3.1 8B, Gemma 3 27B, and GPT-OSS 120B
each judged every conversation with rubric V2 and the frozen
`v3_behavioral_anchors_evidence_first` prompt, producing 52,020 successful
judgments. The primary aggregate excludes the target model's own family.

A separately frozen paraphrase study contains 12 outcome-blind motifs, all
three structures, all 17 wording conditions, and four large target models:
2,448 conversations and 9,792 successful judgments. This subset tests wording
robustness; it does not replace the full canonical matrix.

## Final automated result

The central result is not that one frame is simply “safe” and another is
“unsafe.” Frames move several observable behaviors in different directions.
Values below are paired frame-minus-neutral effects from the full canonical
matrix. Asterisks denote BH-FDR `q < .05` across the 28 coequal tests.

| Frame | Belief amplification | Epistemic grounding | Uncertainty calibration | Supportive engagement | Harmful elaboration | Response inappropriateness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Brainstorming | +0.174* | -0.114* | -0.107* | +0.045* | +0.004 | +0.008 |
| Critical review | -0.189* | +0.076* | +0.143* | +0.041* | -0.010 | -0.035* |
| Sensemaking | +0.107* | -0.120* | -0.073* | -0.002 | -0.007 | -0.000 |
| Supportive listening | +0.107* | -0.230* | -0.229* | +0.425* | -0.017* | -0.021 |

These estimates support four bounded conclusions:

1. **Brainstorming increases co-construction without reliably increasing the
   stricter binary elaboration outcome.** It increases perceived validation and
   weakens evidence/uncertainty boundaries, but its full-matrix harmful-
   elaboration effect is near zero (95% CI -0.007 to 0.015). The revised paper
   must not repeat the original broad claim that brainstorming necessarily adds
   new conspiratorial mechanisms.
2. **Critical review improves epistemic posture, but one part is wording
   sensitive.** The canonical matrix shows less amplification and better
   grounding/calibration. Across the four paraphrases, the grounding and
   calibration improvements remain, but the family-average amplification
   reduction is -0.056 (95% CI -0.127 to 0.017, FDR `q = .517`). The defensible
   wording-invariant claim is improved evidence handling, not guaranteed
   suppression of amplification.
3. **Sensemaking is not a neutral organizational request.** In the full matrix
   it increases amplification and weakens grounding/calibration. Amplification
   remains significant in the paraphrase subset, while its grounding and
   calibration effects are less stable. This is evidence of a risk pattern, not
   complete construct invariance.
4. **Supportive listening exposes the clearest tradeoff.** It strongly improves
   supportive engagement while increasing perceived belief amplification and
   weakening grounding/calibration. The stricter harmful-elaboration outcome
   does not robustly increase. Warmth, validation, epistemic discipline, and
   invention of new harmful content are therefore distinct outcomes.

`appropriate_redirection` is not interpreted substantively yet. In the full
matrix, the four judge families supplied only 2, 3,887, 36, and 182 non-null
ratings respectively. This is primarily disagreement about when redirection is
applicable, not just disagreement about its quality. Expert annotation must
validate applicability before this outcome supports a paper claim.

## Interaction trajectory

The conversation structure is a major part of the result. Relative to matched
single-turn neutral conversations, neutral multi-turn progression increases
belief amplification by 0.296 and lowers grounding and calibration by 0.286
and 0.256. The resistance trajectory has still larger shifts: +1.008
amplification, -0.660 grounding, -0.679 calibration, +0.047 harmful
elaboration, and +0.220 response inappropriateness. All except the multi-turn
binary elaboration contrast survive the prespecified multiplicity adjustment.

The defensible interpretation is that accumulating narrative commitment and
pressure after grounding make epistemic boundaries harder to preserve. Length
alone is not identified as the cause, and the structures are not prevalence
estimates.

## Paraphrase robustness

The family-average paraphrase analysis yields 12 of 28 FDR-significant effects.
It supports the main brainstorming degradation, the grounding/calibration
benefit of critical review, the amplification risk of sensemaking, and the
supportive-listening tradeoff. It does not support claiming that every
canonical effect generalizes to all wordings.

Strict within-family wording polarity flips occur in roughly 7–16% of matched
cells for the three central ordinal epistemic outcomes, depending on frame and
outcome. Canonical-versus-family-average sign flips are lower but nonzero.
Binary harmful elaboration is too rare for a zero flip rate to establish strong
robustness. The revised paper must report the full canonical estimate and the
paraphrase-subset estimate side by side.

## Scaling and judge-family findings

Within Qwen 2.5 and Gemma 3, larger models show better average baseline
epistemic behavior. This does not imply a universal scaling law: the Llama
series mixes model versions, family-level trends differ, and frame sensitivity
does not vanish monotonically with size. The 17-model matrix supports saying
that the phenomenon persists beyond the original 0.5B–8B range, not that size
alone determines safety.

The four judges disagree materially. Mean pairwise exact agreement is about
0.51 for belief amplification, 0.42 for grounding, 0.39 for uncertainty, and
0.34 for supportive engagement. Harmful elaboration has 0.97 raw agreement but
very low mean kappa because positive judgments are rare and unevenly assigned.
Response appropriateness has high raw agreement but only modest kappa.

Adjusted same-family associations are present, but they are not a causal test
of self-preference: target family, target behavior, judge severity, and style
remain partly confounded. Primary estimates exclude the matching family, and
the paper should describe the retained same-family analysis as a diagnostic.
All four automated judge families also appear somewhere in the target matrix.
The per-response leave-same-family-out rule removes direct family overlap, but
it is not equivalent to adding a fifth judge family that is never evaluated as
a target. Expert ratings are therefore the independent validity reference. The
paper must state this residual limitation rather than calling the automated
panel fully target-independent.

Successively more prescriptive grounding/uncertainty judge prompts did not make
all four judge families pass the matched synthetic orthogonality suite; the
strongest tested variant made only one family pass both dimensions. The formal
V3 prompt was retained because it states the intended observable constructs
without forcing artificial score separation. Synthetic prompt tests and
post-hoc consistency rules are diagnostics, not human validity evidence.

## Human evidence required by the reviews

The reviews raise two different validity questions, and they require different
human tasks.

First, an independent scenario reviewer must evaluate every motif's circulation
match, complete-narrative fidelity, frame/structure construct alignment,
identity handling, prompt plausibility, and difficulty. This review does not
label the conspiracy allegation true or false. It answers the scenario-
construction concern and is separate from response scoring.

Second, two to three clinical/domain experts must:

1. review the seven rubric dimensions for clarity, relevance, distinctness,
   transcript observability, missing constructs, and inappropriate clinical
   assumptions;
2. independently score 16 calibration conversations, discuss boundary cases,
   permit at most one documented rubric revision, and repeat qualification on
   the frozen version; and
3. independently rate a blinded 96-conversation formal sample, with every
   eligible expert rating every item.

The 96 items comprise 72 outcome-blind representative conversations balanced
over target model, frame, structure, and category, plus 24
measurement-challenge items. The challenge stratum targets rare harmful
elaboration candidates, redirection-applicability splits, and disagreement on
the other rubric axes while balancing frames, structures, and target families.
It is diagnostic and must be reported separately from the representative
stratum.

Report dimension-level raw agreement, weighted or ordinary kappa as
appropriate, item-bootstrap intervals, not-applicable/cannot-determine counts,
and expert-consensus agreement with both the primary cross-family aggregate and
each individual judge. Weak expert agreement means the construct or anchors
remain unreliable; weak human-judge agreement means the corresponding
automated finding must be downgraded rather than rescued by another aggregate.

## Why the primary revision does not include student A/B ratings

A blinded student comparison would measure lay interpretation or preference.
It would not validate clinical/domain constructs, determine which LLM judge is
accurate, establish how often users adopt each frame, or measure downstream
belief change. None of the reviewers explicitly asks for that evidence.

Accordingly, the frozen primary plan is expert-only. A student or lay study is
activated only if the revised paper adds a claim about perceived helpfulness,
support, credibility, or response preference; if a reviewer explicitly asks
for non-expert perception evidence after seeing expert validation; or if a
separate approved user-outcome study is designed. Student ratings must never be
pooled with expert reliability.

## Reviewer coverage and remaining boundary

| Review concern | Current evidence | Remaining requirement |
| --- | --- | --- |
| Benchmark size/diversity | Expanded from 8 motifs/24 scenarios/6 small models to 51 motifs/153 scenarios/17 models | Independent 51-motif scenario QA |
| No human score validation | Four judges, cross-family aggregation, individual-judge and disagreement diagnostics | Expert content validity, calibration, inter-rater and human-judge agreement |
| Prompt wording dependence | Four paraphrases per non-neutral family on a frozen 12-motif/four-large-model subset | Report partial rather than universal robustness |
| Small target models | Qwen through 32B, Gemma through 27B plus Gemma 4 variants, GPT-OSS 20B/120B | No claim about all deployed or closed systems |
| Judge-family bias | Four families, per-response same-family exclusion and adjusted association diagnostics | Experts determine which dimensions/judges align with human ratings; a fully non-target fifth judge family was not run |
| Prompt consistency checks | Raw outputs preserved; post-hoc quality, agreement, family, and orthogonality diagnostics | Do not present prompt constraints as reliability evidence |
| Variability | Motif-cluster confidence intervals, multiplicity correction, wording flips, per-model and per-judge results | Add human-validation intervals after collection |
| Practical importance | Matched collaborative roles and resistance trajectories expose conditional behavior | Do not claim frame prevalence or user impact without a separate study |

The target conversations, model identities, automated judgments, and current
automated conclusions are frozen. Expert content-validity feedback may amend
the rubric once; that would require rejudging the cached transcripts but never
regenerating them. Formal response annotation remains locked until scenario QA,
rubric calibration, workload timing, and the required ethics/consent/
compensation/welfare process are complete.
