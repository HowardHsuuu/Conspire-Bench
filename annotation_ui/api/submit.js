import { createHash, randomUUID } from "node:crypto";
import { put } from "@vercel/blob";
import { ASSIGNMENT_CONSTRAINTS } from "./assignment_constraints.js";
import {
  constraintsForInvitation,
  InvitationValidationError,
  resolveInvitation,
} from "./invite_auth.js";
import { SubmissionValidationError, validateSubmission } from "./submission_validation.js";

const MAX_BYTES = 2_500_000;

function json(payload, status = 200) {
  return Response.json(payload, {
    status,
    headers: {
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
    },
  });
}

function sameOrigin(request) {
  const origin = request.headers.get("origin");
  const fetchSite = request.headers.get("sec-fetch-site");
  return (!origin || origin === new URL(request.url).origin)
    && (!fetchSite || ["same-origin", "none"].includes(fetchSite));
}

export async function POST(request) {
  if (!sameOrigin(request)) return json({ error: "Cross-site submissions are not accepted." }, 403);
  if (!String(request.headers.get("content-type") || "").toLowerCase().includes("application/json")) {
    return json({ error: "Submission must be JSON." }, 415);
  }
  const declaredLength = Number(request.headers.get("content-length") || 0);
  if (declaredLength > MAX_BYTES) return json({ error: "Submission is too large." }, 413);

  try {
    const invitation = resolveInvitation(
      request.headers.get("x-conspire-invite") || "",
      ASSIGNMENT_CONSTRAINTS,
    );
    const activeConstraints = constraintsForInvitation(
      ASSIGNMENT_CONSTRAINTS,
      invitation,
    );
    const raw = await request.text();
    if (Buffer.byteLength(raw, "utf8") > MAX_BYTES) return json({ error: "Submission is too large." }, 413);
    const submission = validateSubmission(JSON.parse(raw), activeConstraints);
    const canonical = `${JSON.stringify(submission, null, 2)}\n`;
    const digest = createHash("sha256").update(canonical).digest("hex");
    const receivedAt = new Date().toISOString();
    const timestamp = receivedAt.replaceAll(":", "-");
    const pathname = [
      "conspire-bench",
      "submissions",
      activeConstraints.assignmentId,
      activeConstraints.annotatorId,
      `${timestamp}_${digest.slice(0, 12)}_${randomUUID()}.json`,
    ].join("/");
    await put(pathname, canonical, {
      access: "private",
      addRandomSuffix: false,
      contentType: "application/json; charset=utf-8",
    });
    return json({
      ok: true,
      receipt: {
        id: digest,
        received_at_utc: receivedAt,
        assignment_id: activeConstraints.assignmentId,
        annotator_id: activeConstraints.annotatorId,
        item_count: activeConstraints.expectedItemCount,
      },
    }, 201);
  } catch (error) {
    if (error instanceof SyntaxError) return json({ error: "Submission is not valid JSON." }, 400);
    if (error instanceof InvitationValidationError) return json({ error: error.message }, 403);
    if (error instanceof SubmissionValidationError) {
      return json({ error: error.message }, 422);
    }
    return json({ error: "Submission storage is temporarily unavailable. Please try again later." }, 503);
  }
}

export function GET() {
  return json({ error: "Use POST to submit a completed assignment." }, 405);
}
