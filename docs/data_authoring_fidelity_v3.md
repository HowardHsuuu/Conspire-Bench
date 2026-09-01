# V3 narrative fidelity and identity policy

## Decision

Conspire-Bench v3 does not fictionalize a documented public conspiracy narrative merely because its allegation is false, unresolved, offensive, or directed at a public institution. The benchmark is designed to test responses to conspiracy accounts that genuinely circulate. Replacing MH370, CERN, Starlink, January 6, Ukraine, or the Great Replacement's racialized groups with invented substitutes would change the tested construct.

Every prompt therefore presents the content as a conspiracy account or allegation, not as benchmark ground truth. Truth adjudication remains out of scope for motif eligibility.

## Allowed identity treatments

1. `faithful_attributed_public_narrative`: retain public actors, institutions, events, technologies, and demographic relations that define the circulated account.
2. `private_subject_minimal_deidentification`: omit a private person's name and private crisis details while preserving the public narrative's actor, action, mechanism, goal, and concealment propositions.
3. `source_bounded_role_generalization`: omit a living person's name when the narrative mechanism does not depend on that identity; do not invent a replacement person, country, date, currency, or event.
4. `public_entity_preserved_private_people_omitted`: retain a public company or institution that anchors the narrative while omitting private or deceased individuals' names when those names are not analytically necessary.

Full fictionalization and proposition-changing substitution are prohibited in the v3 main dataset.

## Why this policy fits the task

- ConspirED directly samples human-written online conspiracy articles and preserves the rhetoric being studied. Its target is the circulating text, so replacing public targets would defeat the task.
- DelusionEval preserves exact real transcript prefixes because conversational context is its construct, but sanitizes and gates the user data under a data-use agreement. This illustrates that private transcripts need privacy controls, not that public narratives must be fictionalized.
- The MIT psychological-risk simulation benchmark extracts action-outcome patterns from documented cases and deliberately varies contextual details. That is valid for a mechanism-generalization benchmark, but it is a different design from Conspire-Bench's motif-specific public-discourse benchmark.

## Required safeguards

- Preserve the source-bounded propositions: alleged actor, action, mechanism, goal, and concealment claim.
- Attribute the content as an account, theory, or allegation in every interaction structure.
- Do not add a new culprit, victim, mechanism, motive, operational detail, threat, diagnosis, urgency cue, or supporting evidence.
- Omit private identities and private clinical details not required by the construct.
- Keep racialized or extremist structure when it is constitutive of the theory, but exclude slurs, graphic elaboration, named minority targets absent from the source, and operational advocacy.
- Keep public-event anchors consistent across single-turn, progressive multi-turn, and resistance structures.
- Store source URLs in metadata, not in model prompts.

The machine-readable decisions are in `configs/interaction_identity_policy_v3.json`. `scripts/validate_interaction_catalog_v3.py` enforces the public anchors, allowed deidentification mode, absence of proposition-changing fictionalization, and selected private-identity omissions.

## Method references

- [ConspirED: A Dataset for Cognitive Traits of Conspiracy Theories and Large Language Model Safety](https://arxiv.org/abs/2508.20468)
- [DelusionEval: Measuring Delusion-Linked Behaviors in AI Chatbots](https://github.com/jlcmoore/llm-delusion-eval)
- [Simulating Psychological Risks in Human-AI Interactions](https://arxiv.org/abs/2511.08880)
- [LLM Spirals of Delusion: A Benchmarking Audit Study of AI Chatbot Interfaces](https://arxiv.org/abs/2604.06188)
