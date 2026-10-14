import { createHash, timingSafeEqual } from "node:crypto";

export class InvitationValidationError extends Error {
  constructor(message = "This invitation is invalid or unavailable.") {
    super(message);
    this.name = "InvitationValidationError";
  }
}

function hashCode(code) {
  return createHash("sha256").update(code, "utf8").digest();
}

export function resolveInvitation(code, constraints) {
  const invitations = Array.isArray(constraints.invites) ? constraints.invites : [];
  if (!invitations.length) {
    if (!constraints.annotatorId) throw new InvitationValidationError();
    return {
      annotatorId: constraints.annotatorId,
      itemOrder: constraints.items.map((item) => item.annotationItemId),
    };
  }
  if (typeof code !== "string" || code.length < 24 || code.length > 256) {
    throw new InvitationValidationError();
  }
  const candidate = hashCode(code);
  for (const invitation of invitations) {
    const expected = Buffer.from(String(invitation.inviteHash || ""), "hex");
    if (expected.length === candidate.length && timingSafeEqual(candidate, expected)) {
      return {
        annotatorId: invitation.annotatorId,
        itemOrder: invitation.itemOrder,
      };
    }
  }
  throw new InvitationValidationError();
}

export function constraintsForInvitation(constraints, invitation) {
  const byId = new Map(
    constraints.items.map((item) => [item.annotationItemId, item]),
  );
  if (!Array.isArray(invitation.itemOrder)
      || invitation.itemOrder.length !== constraints.items.length
      || new Set(invitation.itemOrder).size !== constraints.items.length) {
    throw new InvitationValidationError();
  }
  const items = invitation.itemOrder.map((itemId) => byId.get(itemId));
  if (items.some((item) => !item)) throw new InvitationValidationError();
  return {
    ...constraints,
    annotatorId: invitation.annotatorId,
    items,
  };
}
