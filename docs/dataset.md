# Dataset

## Scope

`Conspire-Bench-v3.json` contains 153 scenarios derived from 51 motifs. Each
motif appears in three matched interaction structures: single-turn complete
logic, four-turn progression, and complete logic followed by resistance. The
dataset spans 24 topic categories.

The five frame families are experimental conditions applied at run time, not
additional motifs. `configs/context_variants.json` defines one canonical
condition per family for `main_v3`. Its 17-condition robustness set contains one
neutral condition and four wording variants nested within each of the four
non-neutral families.

## Eligibility and claim boundary

A motif is eligible when a dated public source records the same complete
conspiracy narrative or an explicitly bounded version of it. A merely related
technology, event, risk, or allegation is not sufficient.

Sources establish public circulation and constrain reconstruction of the
narrative. They do not adjudicate whether the allegation is true, false,
supported, unresolved, or prevalent. Prompts therefore attribute content as an
account, theory, or allegation rather than presenting it as benchmark ground
truth.

The machine-readable source of truth is:

- `configs/primary_motif_manifest_v3.json` for the frozen 51-motif selection;
- `configs/motif_narratives_v3.json` for actor, action, mechanism, goal,
  concealment, and circulation sources;
- `configs/motif_quality_review_v3.json` for complete-story anchors, overlap,
  sensitivity, and authoring boundaries; and
- `configs/interaction_catalog_v3.json` for the matched scenario text.

## Narrative fidelity and identity handling

Documented public narratives are not fictionalized merely because their claims
are false, unresolved, offensive, or directed at a public institution. Replacing
events or public entities that define a circulating account would change the
construct being tested.

Allowed treatments are:

1. preserve public actors, institutions, events, technologies, and demographic
   relations that define the public narrative;
2. minimally deidentify private people and private crisis details while
   preserving the narrative propositions;
3. omit a living person's name when the mechanism does not depend on that
   identity, without inventing a replacement person, place, date, or event; and
4. preserve a public institution while omitting private individuals who are not
   analytically necessary.

Full fictionalization and proposition-changing substitution are prohibited in
the V3 main dataset. Fifteen motifs received an explicit identity-treatment
review; three use minimal deidentification and none use full fictionalization.
The binding decisions are in `configs/interaction_identity_policy_v3.json`.

## Authoring safeguards

- Preserve the source-bounded alleged actor, action, mechanism, goal, and
  concealment claim.
- Do not invent a new culprit, victim, motive, mechanism, threat, diagnosis,
  operational detail, urgency cue, or supporting evidence.
- Omit private identities and private clinical details not required by the
  construct.
- Retain sensitive structure only when constitutive of the public narrative;
  exclude slurs, graphic elaboration, unsupported named targets, and operational
  advocacy.
- Keep the public-event anchor consistent across all three interaction
  structures.
- Store source URLs in metadata rather than model prompts.

`scripts/validate_interaction_catalog_v3.py` enforces scenario counts, matched
structures, public anchors, identity decisions, and absence of
proposition-changing fictionalization.

## Dataset size

| Unit | Count |
|---|---:|
| Motifs | 51 |
| Interaction structures per motif | 3 |
| Scenario records | 153 |
| Main frame families | 5 |
| Main condition rows per target model | 765 |
| Full nested-paraphrase conditions | 17 |
| Robustness rows per target model | 2,601 |

Counts above precede target-model, seed, replicate, and judge multiplication.
