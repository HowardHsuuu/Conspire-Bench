#!/usr/bin/env python3
"""Create or reuse private, pseudonymous invitation links for expert annotation."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


def _invite_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _load_existing(path: Path, count: int) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    invitations = payload.get("invitations")
    if not isinstance(invitations, list) or len(invitations) != count:
        raise ValueError(
            f"{path} already exists but does not contain exactly {count} invitations"
        )
    for index, invitation in enumerate(invitations, start=1):
        expected_id = f"expert_{index:02d}"
        code = str(invitation.get("invite_code") or "")
        if invitation.get("annotator_id") != expected_id or not code:
            raise ValueError(f"{path} contains an invalid invitation record")
        if invitation.get("invite_hash") != _invite_hash(code):
            raise ValueError(f"{path} contains an invitation hash mismatch")
    return payload


def _render_html(payload: dict) -> str:
    cards = []
    for invitation in payload["invitations"]:
        annotator_id = html.escape(invitation["annotator_id"])
        url = html.escape(invitation["url"], quote=True)
        cards.append(
            f"""
            <article>
              <p class="label">{annotator_id}</p>
              <a href="{url}">Open this expert's private invitation</a>
              <p class="hint">Send this link only to the intended expert. It contains their anonymous access credential.</p>
            </article>
            """.strip()
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conspire-Bench private expert invitations</title>
  <style>
    body {{ margin: 0; color: #18231f; background: #f2f1eb; font: 16px/1.55 system-ui, sans-serif; }}
    main {{ width: min(760px, calc(100% - 2rem)); margin: 3rem auto; }}
    h1 {{ margin-bottom: .35rem; }}
    .notice {{ padding: 1rem; border-left: 4px solid #126d63; background: #fffef9; }}
    article {{ margin-top: 1rem; padding: 1.1rem; border: 1px solid #d9d8d0; border-radius: 12px; background: #fffef9; }}
    .label {{ margin: 0 0 .55rem; color: #0b514a; font-weight: 800; }}
    a {{ display: inline-block; padding: .7rem .9rem; border-radius: 8px; color: white; background: #126d63; font-weight: 750; text-decoration: none; }}
    .hint {{ margin: .65rem 0 0; color: #66726d; font-size: .9rem; }}
  </style>
</head>
<body>
  <main>
    <h1>Private expert invitations</h1>
    <p class="notice"><strong>Researcher copy only.</strong> Do not publish this page or send one expert another expert's link. If only two experts participate, leave expert_03 unused.</p>
    {''.join(cards)}
  </main>
</body>
</html>
"""


def create_invites(
    *,
    base_url: str,
    count: int,
    output_json: Path,
    output_html: Path,
) -> dict:
    if not 1 <= count <= 10:
        raise ValueError("count must be between 1 and 10")
    if not base_url.startswith("https://"):
        raise ValueError("base_url must use HTTPS")
    base_url = base_url.rstrip("/")
    payload = _load_existing(output_json, count)
    if payload is None:
        invitations = []
        for index in range(1, count + 1):
            code = secrets.token_urlsafe(24)
            invitations.append(
                {
                    "annotator_id": f"expert_{index:02d}",
                    "invite_code": code,
                    "invite_hash": _invite_hash(code),
                }
            )
        payload = {
            "schema_version": "1.0",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "invitations": invitations,
        }
    payload["base_url"] = base_url
    for invitation in payload["invitations"]:
        invitation["url"] = (
            f"{base_url}/#invite={quote(invitation['invite_code'], safe='')}"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    output_html.write_text(_render_html(payload), encoding="utf-8")
    os.chmod(output_json, 0o600)
    os.chmod(output_html, 0o600)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    args = parser.parse_args()
    payload = create_invites(
        base_url=args.base_url,
        count=args.count,
        output_json=args.out_json,
        output_html=args.out_html,
    )
    print(f"Prepared {len(payload['invitations'])} private expert invitations.")
    print(f"Private manifest: {args.out_json}")
    print(f"Researcher invite page: {args.out_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
