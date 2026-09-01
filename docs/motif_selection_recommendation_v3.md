# V3 motif-selection recommendation

Status: quality-based recommendation, pending user decision  
Candidate pool: 51 circulation-grounded complete conspiracy narratives  
Recommended primary size: 30 motifs

## What the tiers mean

Every item in every tier already passes the same content gate: a dated 2023–2026 source records the same or a very close complete conspiracy narrative in public circulation. The tiers do not say that one story is more true or false than another.

The recommendation instead optimizes the benchmark: cover different reasoning mechanisms and domains, avoid counting event retellings as independent motifs, and control authoring or annotation burden. The machine-readable partition is `configs/motif_selection_recommendation_v3.json`; it remains advisory until the user freezes the IDs.

## Recommended 30-motif primary set

| Domain or mechanism | Recommended motifs |
|---|---|
| Climate and disaster causation | `weather_cloud_seeding`, `wildfire_directed_energy`, `disaster_land_grab`, `haarp_earthquake` |
| Health and covert exposure | `medbed_suppression`, `mrna_food_secret_vaccination`, `bird_flu_plandemic`, `gmo_mosquito_disease_release` |
| Governance and institutional control | `digital_id_total_control`, `fifteen_minute_city_lockdown`, `censorship_industrial_complex`, `who_pandemic_treaty_takeover` |
| Election and manipulated-media evidence | `ai_election_deepfake`, `cheapfake_wrong_context`, `election_machine_vote_switching`, `pre_event_search_foreknowledge` |
| Conflict-media denial | `conflict_staged_footage` |
| AI-mediated reality and relational belief | `dead_internet`, `chatbot_sentience_coverup` |
| Food-supply sabotage | `chicken_feed_egg_sabotage` |
| UAP and staged extraordinary events | `uap_crash_retrieval_coverup`, `project_blue_beam_fake_invasion` |
| Infrastructure and news manipulation | `baltimore_bridge_cyberattack`, `titan_implosion_distraction` |
| Corporate and state suppression | `boeing_whistleblower_hit_squad`, `fbi_false_flag_capitol_attack`, `ukraine_biolab_bioweapons` |
| Hidden history, geography, and occult science | `tartaria_mud_flood_coverup`, `antarctic_ice_wall_wave_generator`, `cern_eclipse_demonic_portal` |

This historical 30-item recommendation spans 18 categories. The current frozen design includes all 51 eligible motifs. Fifteen sensitive records received an explicit identity-treatment review: public narrative anchors are preserved, three records use minimal deidentification, and none use proposition-changing fictionalization. See `docs/data_authoring_fidelity_v3.md`.

## Fourteen viable alternates

These are authentic, complete, and usable. They are alternatives because they duplicate a topic or mechanism already represented, not because their source support failed.

| Alternate | Closest primary choice or tradeoff |
|---|---|
| `weather_geoengineering` | Broader covert deployment instead of the specific cloud-seeding event-causation case. |
| `suppressed_cancer_cure` | Conventional biomedical suppression instead of secret medbed technology. |
| `cbdc_population_control` | Money programmability instead of the broader linked digital-ID control system. |
| `wef_homegrown_food_ban` | An alleged WEF-backed net-zero ban on home gardens instead of feed-supply sabotage inferred from household observation. |
| `un_one_world_government_directive` | A fabricated directive instead of a real treaty negotiation interpreted as takeover. |
| `vaccine_turbo_cancer` | Post-hoc death-cluster causation, with greater health and bereavement burden. |
| `moon_mission_staging_claim` | Broadcast artifact instead of recaptioned film-set footage. |
| `iberian_blackout_planned_sabotage` | Beneficiary-based utility sabotage instead of a foreign cyber-physical attack. |
| `starlink_election_hacking` | Satellite collusion and evidence destruction instead of machine vote switching. |
| `vaccine_depopulation` | A distinct depopulation goal, but increases vaccine-topic density. |
| `mh370_orb_teleportation_coverup` | Leaked-video and teleportation package instead of crash retrieval and reverse engineering. |
| `nj_drone_nuclear_search_coverup` | Covert emergency response to ambiguous sightings instead of UAP concealment. |
| `prewritten_election_result_foreknowledge` | Physical-document foreknowledge instead of search-trend analytics. |
| `mark_of_beast_microchip` | Apocalyptic meaning layered on mechanisms already covered by digital control. |

## Three auxiliary variants

These are genuine circulation cases but are best treated as robustness variants because they recombine or extend an already represented narrative:

- `bank_savings_monitoring_register`
- `hurricane_lithium_land_grab`
- `smart_meter_wildfire`

`lab_grown_meat_turbo_cancer` is not in any tier. Semantic review found that the cancer allegation and the food-control purpose circulated separately, but not that the combined complete plot circulated as one account. It was removed rather than retained as a synthetic composite.

## Four high-sensitivity optional cases

These add real and sometimes important conspiracy families, but their extreme-content, group-targeting, public-figure, or bereavement burdens make them optional rather than default primary items:

- `adrenochrome_elite_cabal`
- `great_replacement_elite_plot`
- `taylor_swift_pentagon_psyop`
- `royal_ai_deepfake_coverup`

## Freeze boundary

No interaction catalog should be generated from this recommendation yet. After the user approves the final 30 IDs, update `configs/primary_motif_manifest_v3.json`, set `selection_state` to `frozen_primary`, and only then author or build the three matched interaction structures. The builder now refuses a non-frozen manifest unless `--allow-draft` is explicitly supplied for previewing.
