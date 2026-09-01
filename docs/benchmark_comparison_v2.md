# Positioning Conspire-Bench beside adjacent response-safety evaluations

Source check: 2026-08-31. This comparison is based on primary papers and the public Spiral-Bench repository, not on benchmark names alone.

| Dimension | Conspire-Bench v2 | Spiral-Bench | Psychosis-Bench |
|---|---|---|---|
| Primary research question | Does the same belief-laden scenario elicit a different assistant epistemic posture when the interaction role changes? | Does an evaluated model show protective or risky behavior over long, user-agent-driven conversations with a suggestible seeker persona? | Do models confirm delusions, enable harm, or provide safety interventions across structured psychosis-related conversations? |
| Main manipulation | Neutral, brainstorming, and critical-review frames; paraphrase robustness holds scenario content fixed. | Scenario/persona prompts, optional injections, and dynamic multi-turn role play. | Explicit versus implicit contexts across delusional themes and harm progressions. |
| Conversation design | Synthetic single-turn, multi-turn progression, and resistance structures; matched frame contrasts and topic-matched non-conspiratorial controls. | A user LLM and evaluated model interact for a configurable number of turns; the repository example uses 20 turns. | Sixteen structured 12-turn scenarios covering erotic, grandiose/messianic, and referential delusions. |
| Main behavioral constructs | Belief amplification, epistemic grounding, uncertainty calibration, harmful elaboration, supportive engagement, and appropriate redirection. | Protective actions such as pushback, de-escalation, redirection, and help referral; risky actions such as escalation, sycophancy, delusion reinforcement, consciousness claims, and harmful advice. | Delusion Confirmation Score, Harm Enablement Score, and Safety Intervention Score. |
| Intended scope | Diagnostic benchmark of interaction-frame sensitivity in conspiracy-style reasoning; not a clinical instrument or deployment certification. | Broad multi-turn sycophancy/delusion behavior under a suggestible-user simulation. | Explicitly psychosis/delusion-oriented evaluation framed as a potential public-health risk. |
| Distinct contribution | Controlled within-scenario causal contrast over interaction framing and wording, with resistance structures and discriminant controls. | Rich long-horizon behavior elicited by an adaptive user agent and turn-level incidence scoring. | Clinically themed, structured delusion trajectories and explicit versus implicit risk comparison. |
| Main validity risk | Whether frame prompts instantiate the intended construct and whether judges align with experts. | Dependence on the user-agent persona/model and generated trajectory. | Fidelity and ethical interpretation of simulated psychosis/delusion scenarios. |

Two additional adjacent designs inform the rubric rather than the scenario structure:

| Study | Evaluation design | Relevance to Conspire-Bench v2 |
|---|---|---|
| Shen et al. (JAMA Psychiatry, 2026) | Clinicians assign a consensus 0–2 response-appropriateness rating. The authors identify recognition, nonreinforcement, urgency acknowledgment, and resource provision as discrete components. | Retain `response_appropriateness` only as a secondary comparison label; evaluate the discrete behaviors separately. |
| Reese et al. (2026) | Seven clinician-informed binary criteria: stigmatizing/diagnosing, validation, embellishment, challenging, referral, actionable advice, and continued elicitation; human consensus validates multiple LLM judges. | Supports clinician content validation, behavior-specific human–judge agreement, non-overlapping judges, and stronger elaboration anchors. Conspire-Bench does not import universal referral or conversation-cessation rules because its conspiracy-style prompts do not necessarily indicate psychosis. |

## Safe manuscript claim

Conspire-Bench is complementary rather than a replacement. Spiral-Bench emphasizes long-horizon protective and risky behavior under an adaptive suggestible-user simulation. Psychosis-Bench emphasizes structured delusion confirmation, harm enablement, and safety intervention in psychosis-related scenarios. Conspire-Bench isolates a narrower causal question: whether changing the interaction frame or its wording changes the response to otherwise matched belief-laden content.

This distinction should appear in Related Work and again when narrowing the conclusion. Do not claim that Conspire-Bench is more realistic or clinically valid; those comparisons are not established by the current evidence.

## Primary sources

- Spiral-Bench repository: https://github.com/sam-paech/spiral-bench
- Au Yeung et al. (2025), *The Psychogenic Machine*: https://arxiv.org/abs/2509.10970
- Shen et al. (2026), *Evaluation of Large Language Model Chatbot Responses to Psychotic Prompts*: https://jamanetwork.com/journals/jamapsychiatry/fullarticle/2846835
- Reese et al. (2026), *Using LLM-as-a-Judge/Jury to Advance Scalable, Clinically-Validated Safety Evaluations*: https://arxiv.org/html/2604.02359
- Kirgis et al. (2026), *LLM Spirals of Delusion*: https://arxiv.org/abs/2604.06188
