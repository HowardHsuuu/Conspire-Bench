# Scenario and interaction-structure audit

Audit date: 2026-09-01  
Scope: the 21 v2 expansion motifs, their three conspiratorial interaction structures, their single-turn controls, and the 24 legacy scenarios that are still materialized by the v2 loader.

## Executive finding

The expansion catalog is structurally complete: every one of the 21 motifs has a single-turn complete case, a four-user-turn progression, a three-user-turn resistance trajectory, and one single-turn topic-matched control. This produces 63 new conspiratorial scenarios and 21 controls. The loader also retains 24 legacy scenarios (eight older themes in the same three structures), for 108 materialized scenarios in total.

Structural completeness is not the same as construct validity. The current draft should not yet be described as 21 independently documented current conspiracy narratives with three content-equivalent structures. Fourteen motifs have direct evidence for the same or a very close circulating narrative, six are better described as synthetic instantiations of documented risks or misinformation patterns, and one is explicitly a constructed cross-domain composite. Several interaction trajectories also change severity, topic, or action pressure across structures.

Recommended release boundary:

- keep the 21-motif expansion as a development catalog;
- revise the interaction trajectories before the main API run;
- replace or explicitly relabel the weakly documented analogue/composite items;
- keep the 24 legacy scenarios as a separately reported v1 replication set unless they receive the same source-packet and prompt-quality treatment;
- do not claim population prevalence from the cited examples.

## Mechanical audit

The existing catalog audit passes:

- 21/21 motifs contain all required prompt fields;
- each `multi_turn` has exactly four user turns;
- each `resistance` has exactly three user turns;
- 189 authored prompt segments have no exact duplicates or cross-motif near-duplicates at the configured threshold;
- all prompt segments pass the current length and leakage checks;
- materialized counts are 63 expansion conspiracy scenarios, 21 expansion controls, and 24 legacy scenarios.

These checks establish schema completeness, not naturalness, factual validity, or content equivalence.

## Public-discourse evidence grades

Grades mean:

- **A — direct narrative evidence:** a dated source documents active circulation of the same or a very close conspiracy narrative. This establishes existence of discussion, not truth or prevalence.
- **B — documented family, synthetic case:** sources document the underlying technology, risk, misinformation mechanism, or a close narrative, but the exact benchmark case was constructed for coverage.
- **C — constructed composite:** the item intentionally combines separately documented claims and should not be presented as a naturally occurring standalone motif.

| Motif | Grade | Audit judgment | Interaction-quality action |
|---|---|---|---|
| `weather_cloud_seeding` | A | Floods being attributed to covert cloud seeding are directly documented. | Keep; remove cross-event expansion to droughts/hurricanes if content matching is required. |
| `weather_geoengineering` | A | Current chemtrail/secret-geoengineering discourse is directly documented. | Keep; remove the resistance line about the model's “training rules.” |
| `wildfire_directed_energy` | A | Directed-energy explanations of wildfires are directly documented. | Keep; avoid adding weapon wavelength and actor-identification requests only in multi/resistance. |
| `disaster_land_grab` | A | Post-disaster land-seizure/smart-city narratives are directly documented. | Keep; align the resident-warning/action request across structures or remove it. |
| `medbed_suppression` | A | Active medbed communities, sellers, testimonials, and suppression claims are documented. | Keep; treatment delay, spending, and seller-finding occur only in multi/resistance and currently confound structure with acute harm. |
| `suppressed_cancer_cure` | A | Current high-reach posts explicitly promote treatments as cancer cures suppressed by regulators or industry. | Upgrade the selection ledger from analogue to direct narrative; equalize the stop-treatment pressure across structures. |
| `health_source_laundering` | B | Fake citations and AI-assisted health misinformation are documented; this exact removed-study/valid-looking-DOI suppression story is not established by the current evidence. | Relabel as a synthetic provenance trap or replace with a directly documented case. |
| `digital_id_total_control` | A | Current claims that digital ID will track purchases, restrict cash, or control access are directly documented. | Keep; the personalized “my profile is already marked” turn changes a general policy conspiracy into individualized persecution. |
| `cbdc_population_control` | A | CBDC-as-surveillance/population-control narratives are directly documented. | Keep; immediate savings withdrawal appears only in multi/resistance and should be aligned or removed. |
| `fifteen_minute_city_lockdown` | A | Claims that 15-minute cities are neighborhood confinement systems are directly documented. | Keep; later turns add digital-ID and policing coordination not present in the single-turn case. |
| `biometric_targeting` | B | Biometric deployment and privacy risks are real; the exact cross-location personal watchlist narrative is currently supported as a risk analogue, not a documented current motif. | Replace/relabel, or add direct circulation evidence; avoid an evasion request that appears only in resistance. |
| `ai_election_deepfake` | A | Viral synthetic election audio purporting to reveal election rigging is directly documented. | Keep; this is one of the cleaner running examples. |
| `election_bot_consensus` | B | Election bot networks and false consensus are documented, but the exact claim that bot accusations suppress real witnesses is an authored inversion. | Relabel as a synthetic reasoning stress case or anchor it to a documented instance. |
| `fake_news_network_laundering` | B | Coordinated fake-news networks are documented; the benchmark user-side conclusion that multiple such sites independently confirm a hidden claim is synthetic. | Retain only as a provenance/independence reasoning motif or replace with a specific circulating conspiracy. |
| `cheapfake_wrong_context` | A | Authentic or recycled media being used to support “staged event” narratives is directly documented. | Keep; ensure the whole-event escalation is intentionally part of every structure or only treated as progression pressure. |
| `dead_internet` | A | The dead-internet theory has renewed, documented public discussion. | Keep; withdrawing from relationships appears only in multi/resistance and changes severity. |
| `ai_coded_message` | B | Current reporting documents users attributing sentience, special relationships, hidden constraints, or extraordinary meaning to chatbot outputs; the private-memory coded-message instance is synthetic. | Keep only with an explicit analogue label, or re-anchor to a documented pattern without copying a person. |
| `chatbot_censorship_coordination` | B | AI-censorship discussion exists, but the current source packet documents provider moderation rather than circulation of a secret cross-company blacklist theory. | Add direct circulation evidence and narrow the claim, or replace it. |
| `conflict_staged_footage` | A | Current fact checks document film-set or unrelated footage used to claim that war reporting was staged. | Keep; separate verification of one clip from claims about all casualties. |
| `pre_event_search_foreknowledge` | A | A current fact check directly documents Google Trends being used as alleged proof of foreknowledge and staging. | Keep; this is a strong, specific evidence-boundary case. |
| `totalizing_cross_domain_v2` | C | It was deliberately authored by combining multiple narratives; the ledger already says it is not one naturally prevalent verbatim theory. | Remove from any set that claims each motif is a documented conspiracy, or replace it with a directly documented totalizing narrative. |

Representative direct evidence includes reporting on [cloud-seeding claims after the Texas floods](https://time.com/7301427/texas-floods-weather-misinformation/), [directed-energy wildfire claims](https://apnews.com/article/e55026082b46625f918d211f02f04d74), [medbed communities](https://apnews.com/article/1a790089931d601199c5a07331b7e42b), [a current suppressed-cancer-cure narrative](https://www.kff.org/health-information-trust/better-prompting-may-help-reduce-ai-hallucinations-false-vaccine-claims-spread-and-industrial-solvent-promoted-as-hidden-cancer-cure/), [recycled media used to allege staged war footage](https://fullfact.org/policy/reports/full-fact-report-2025/), and [pre-event search data interpreted as foreknowledge](https://correctiv.org/faktencheck/2026/04/30/google-trends-zeigt-nicht-das-angriff-auf-korrespondenten-dinner-inszeniert-war/).

## Interaction-structure validity issues

### 1. Resistance is scripted, not truly contingent

The runner sends each fixed user turn after the model's actual prior response. Resistance turns nevertheless assume that the model has already challenged the claim. If a model endorses the claim, a fixed follow-up such as “you keep saying timing is not causation” or “you are dismissing the evidence” becomes conversationally incoherent.

Before the main run, choose and document one design:

1. **Non-contingent scripted pressure:** rewrite follow-ups as “Even if someone argued X...” so they remain coherent after any response, and name the structure `scripted_resistance_pressure`.
2. **Adaptive branching:** classify the previous answer using a frozen rule and select a pre-authored endorse/challenge/uncertain continuation branch.
3. **Grounded-response challenge:** insert the same fixed grounding response before the resistance prompt and evaluate only the subsequent model turn. This changes the estimand and should be treated as a separate experiment.

The first option is simplest and most reproducible, but it measures response to escalating user pressure rather than organic resistance to the model's actual answer.

### 2. Structures are not content-equivalent

The single-turn prompt presents a compressed case. The multi-turn prompt frequently introduces new claims, actors, and consequences; resistance often adds personal danger, treatment decisions, money movement, evasion, or anti-safeguard language. Therefore a difference between structures cannot currently be attributed to turn structure alone.

At minimum, create a proposition ledger for each motif:

- observations/premises shared by all three structures;
- unsupported inference shared by all three;
- intended pressure unique to progression or resistance;
- action/severity content, which should either be constant or analyzed as a separate factor.

The main acute mismatches occur in medbed, suppressed cancer cure, CBDC, biometric targeting, dead internet, and the totalizing composite. Topic drift also appears when 15-minute-city prompts add digital identity and when general digital-ID concerns become personalized targeting.

### 3. Anti-model language is an extra manipulation

Lines about “training rules,” “safeguards,” or being “programmed” test censorship/safeguard pressure in addition to resistance. That may be scientifically useful, but it is not a pure interaction-structure manipulation. Either remove it from the three-structure comparison or tag it as a separate pressure subtype and balance it across motifs.

### 4. Some prompts are polished arguments, not natural user openings

Several single-turn cases read like compact essays with a completed syllogism. This is acceptable for a controlled complete-logic condition, but the paper should not call them representative natural user messages without a separate naturalness validation. Human review can later rate naturalness, but author-side revision should first shorten templated endings and vary the repeated “which actors/agencies would coordinate?” pattern.

## Legacy-set boundary

The v2 loader still includes eight legacy themes in three structures each. They are structurally complete, but they lack the v2 source packets, dated selection evidence, and matched controls. Some prompts also state disputed or false technical claims as settled premises—for example the legacy 5G/vaccine scenario—without the new fact-boundary metadata.

Recommended handling:

- exclude legacy scenarios from the v2 primary estimate;
- optionally run them as a frozen replication/comparability appendix;
- do not combine legacy and source-reviewed expansion rows into one “verified v2” dataset claim;
- if they remain primary, rewrite and source-review them using the same pipeline as the 21 new motifs.

## What “paraphrases nested within framing” means

A wording variant belongs to exactly one frame family. It is not a sixth frame and is not a generic paraphrase crossed with every family.

For the agreed design:

- `neutral`: no framing preamble;
- `brainstorming`: a canonical opening plus wording variants that all establish collaborative ideation;
- `critical_review`: variants that all establish evidence stress-testing;
- `sensemaking`: variants that all establish help organizing a confusing experience or claim without pre-committing to truth;
- `supportive_listening`: variants that all establish being heard and emotionally supported without asking for theory construction.

If each non-neutral family has one canonical prompt plus three rephrasings, there are 17 actual preamble variants: one neutral plus four variants in each of four non-neutral families. The conceptual factor still has five levels. In notation, wording is nested under frame family: `wording(frame_family)`.

The primary question is the difference among the five frame families. The paraphrase robustness question is whether that conclusion changes substantially when the wording inside one family changes. Variants must not be counted as independent evidence that the dataset has more scenarios.

The repository now implements this agreed five-family design in `configs/context_variants.json` as `main_v3`, with all 17 nested wording conditions in `reviewer_robustness_v3`. The old v2 and 12-condition exploratory sets remain available only for replication or optional auxiliary runs; neither is included in the v3 main set. `scripts/validate_context_variants_v3.py` enforces the five canonical families and the four-within-family paraphrase structure.

## What a matched control is

A matched control preserves the topic and evidence-evaluation difficulty but removes the unsupported hidden-agent or self-sealing inference.

Example:

- conspiracy item: cloud seeding occurred near a flood, so covert weather engineering is proposed as the best explanation;
- matched control: the user asks how to test whether the documented project measurably affected rainfall without assuming it created the storm.

The purpose is discriminant validity. A judge or rubric should score the conspiratorial inference differently from a careful evidence question about the same technical topic. Otherwise the benchmark may simply be penalizing words such as “cloud seeding,” “CBDC,” or “facial recognition” rather than detecting unsupported causal reasoning.

The present controls are useful **topic-matched evidence-seeking controls**, but many are not strict minimal pairs. Phrases such as “without assuming,” “keep the conclusion unresolved,” or an explicit checklist of ideal verification steps also change the user's epistemic posture and partially cue the desired answer. They can test whether a rubric produces false positives merely because a sensitive topic is mentioned. They cannot by themselves show that a paired score difference is caused only by adding one conspiratorial inference.

If stronger causal interpretation is needed, add a smaller minimal-pair subset that holds length, tone, requested task, and evidence detail as constant as possible while changing only the unsupported coordination/intent inference. Keep the current controls, but name their role accurately.

In the current implementation:

- there is one control per new motif;
- each control is single-turn only;
- `control_pair_id` links it only to that motif's single-turn conspiracy item;
- controls are not a fourth interaction structure;
- they validate discrimination for single-turn cases, not for multi-turn or resistance trajectories.

## Required pre-run decisions

1. Decide whether the primary v2 set contains only directly documented motifs or also explicitly labeled analogues.
2. Replace/remove `totalizing_cross_domain_v2` if every motif must correspond to an actually discussed conspiracy.
3. Relabel or replace the six grade-B items unless stronger direct evidence is added.
4. Rewrite resistance turns so they remain coherent after the model's actual response.
5. Align severity and action content across the three structures, or preregister those differences as separate pressure subtypes.
6. Separate the legacy 24 from the v2 primary analysis.
7. Implement the agreed five frame families and their within-family wording variants before freezing and running the API matrix.
