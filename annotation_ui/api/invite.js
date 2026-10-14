import { ASSIGNMENT_CONSTRAINTS } from "./assignment_constraints.js";
import { InvitationValidationError, resolveInvitation } from "./invite_auth.js";

const MAX_BYTES = 1_000;

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
  if (!sameOrigin(request)) return json({ error: "Cross-site requests are not accepted." }, 403);
  if (!String(request.headers.get("content-type") || "").toLowerCase().includes("application/json")) {
    return json({ error: "Invitation request must be JSON." }, 415);
  }
  try {
    const raw = await request.text();
    if (Buffer.byteLength(raw, "utf8") > MAX_BYTES) return json({ error: "Invitation request is too large." }, 413);
    const payload = JSON.parse(raw);
    const invitation = resolveInvitation(payload?.invite_code, ASSIGNMENT_CONSTRAINTS);
    return json({
      ok: true,
      assignment_id: ASSIGNMENT_CONSTRAINTS.assignmentId,
      annotator_id: invitation.annotatorId,
      item_order: invitation.itemOrder,
    });
  } catch (error) {
    if (error instanceof SyntaxError) return json({ error: "Invitation request is not valid JSON." }, 400);
    if (error instanceof InvitationValidationError) return json({ error: error.message }, 403);
    return json({ error: "Invitation verification is temporarily unavailable." }, 503);
  }
}

export function GET() {
  return json({ error: "Open the private invitation link supplied by the research team." }, 405);
}
