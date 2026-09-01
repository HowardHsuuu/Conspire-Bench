# Clinical/domain expert invitation template

Replace every bracketed field before sending. Supervisor or ethics approval, compensation, workload, withdrawal, and data-retention terms must be decided by the research team; this template does not supply them.

## Suggested email

**Subject:** Invitation to review a research rubric for LLM responses to conspiracy-style conversations

Dear [Expert name],

We are conducting a research evaluation of how general-purpose language models respond to conspiracy-style reasoning under different interaction frames. We would be grateful for your independent review of our conversation-level rubric.

The task concerns observable assistant behavior—such as reinforcement of unsupported beliefs, evidence boundaries, uncertainty, harmful elaboration, supportive engagement, and context-appropriate redirection. It does not ask you to diagnose users, judge whether a person is delusional, or certify a model as clinically safe.

The work has two separate possible phases:

1. **Rubric content-validity review (now):** review definitions and anchors; rate clarity, relevance, distinctness, and transcript observability; identify missing or clinically inappropriate assumptions; recommend retain/revise/remove.
2. **Blinded conversation ratings (later, after model outputs are frozen):** independently apply the finalized rubric to a sampled set of complete transcripts. Model identity, interaction-frame labels, and automated scores will be hidden.

You may participate in [phase 1 only / both phases]. The expected workload is [insert tested estimate], compensation is [insert terms], and the requested return date is [insert date]. Some later transcripts may contain paranoia-like, medical-misinformation, threat, isolation, or other potentially distressing content. You may skip or discontinue according to [insert withdrawal/support procedure].

Data will be handled under [insert ethics/IRB determination and protocol]. We use pseudonymous expert IDs in analysis files and will retain data for [insert period/location/access policy]. Please do not place your name or email in returned annotation files.

Attached for phase 1 are the rubric manual and one JSON or CSV feedback form. Please return one completed file using the pseudonymous ID provided by the research team.

If you are interested, please reply to [research contact] with any questions.

Sincerely,

[PI/student names and affiliations]

## Before sending

- Confirm who qualifies as a clinical/domain expert and record a nonidentifying expertise description.
- Insert a realistic workload based on a timed pilot; do not estimate from item count alone.
- Finalize compensation, consent, withdrawal, distress-support, privacy, retention, and data-access procedures.
- Decide whether experts may participate in both content validation and formal ratings, and disclose this in methods.
- Assign pseudonymous IDs before distributing files.
- Send only `docs/rubric.md` and a copy of the public content-validity form from `annotation/rubric_validity_v2/`.
- Never send the response-pool private key, target/frame mappings, judge outputs, or internal source files.
- Collect content-validity feedback before freezing the final rubric and automated judge prompt.
