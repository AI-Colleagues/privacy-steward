"""Generate synthetic PII-rich corpus for throughput benchmarking.

Produces 10 small .txt files in benchmarks/data/, each containing ~10 emails.
"""

from __future__ import annotations

import random
import string
from pathlib import Path


NAMES = [
    "Alice Johnson", "Bob Martinez", "Carol Chen", "David Kim",
    "Eva Müller", "Frank Okafor", "Grace Tanaka", "Henry Dubois",
    "Isabel Santos", "James O'Brien", "Karen Patel", "Liam Nguyen",
]
EMAILS = [
    "alice.j@acme.com", "bob.martinez@example.org", "c.chen@corp.io",
    "david.k@startup.ai", "eva.m@firma.de", "f.okafor@uni.edu",
    "g.tanaka@techco.jp", "h.dubois@lab.fr", "i.santos@uni.pt",
    "james.ob@company.ie",
]
PHONES = [
    "+1-555-867-5309", "800-555-0199", "(415) 555-2671",
    "+44 20 7946 0958", "+49 30 12345678",
]
ORGS = [
    "Acme Corp", "TechFirm Inc", "DataSoft GmbH", "InnoLab Ltd",
    "StartupXYZ", "Global Analytics LLC",
]
ADDRESSES = [
    "42 Maple Street, Springfield, IL 62701",
    "100 Innovation Drive, San Francisco, CA 94103",
    "9 Oxford Road, London EC1A 1BB",
]


def _random_email() -> str:
    """Return one synthetic email with PII spanning ~6 lines."""
    name = random.choice(NAMES)
    org = random.choice(ORGS)
    email = random.choice(EMAILS)
    phone = random.choice(PHONES)
    addr = random.choice(ADDRESSES)
    acct = "-".join("".join(random.choices(string.digits, k=4)) for _ in range(4))
    return (
        f"Dear {name},\n\n"
        f"Thank you for your interest in {org}.\n"
        f"You can reach us at {email} or call {phone}.\n"
        f"Our office is located at {addr}.\n"
        f"Your reference number is {acct}.\n\n"
        f"Best regards,\n"
        f"{random.choice(NAMES)}"
    )


def generate_file(dest: Path, n_emails: int = 10) -> None:
    """Write *n_emails* synthetic PII emails to *dest*."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    content = "\n\n---\n\n".join(_random_email() for _ in range(n_emails))
    dest.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    random.seed(42)
    out = Path(__file__).parent / "data"
    # Remove old large files if present
    for old in out.glob("corpus_*.txt"):
        old.unlink()
    for i in range(1, 11):
        generate_file(out / f"batch_{i:02d}.txt", n_emails=10)
    print(f"Generated 10 benchmark files in {out}/")
    for f in sorted(out.glob("*.txt")):
        print(f"  {f.name}: {f.stat().st_size:,} bytes")
