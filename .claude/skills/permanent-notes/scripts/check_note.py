"""Mechanical checks for a permanent note.

Usage: python check_note.py permanent-notes/03-topic/Notes.md

Checks relative links and #anchors (kramdown ids), GitHub blob links to files in
this repo, $$ balance, leftover image references and front matter. Also points at
tutor-sounding phrases for a second look. Exits 1 if anything needs fixing.
"""

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
HEADER = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
GITHUB_BLOB = re.compile(r"^https://github\.com/[^/]+/[^/]+/blob/[^/]+/(.+)$")
TUTOR_PHRASES = [
    "note that", "notice", "let's", "let us", "recall", "remember that",
    "you might", "as we'll see", "as we will see", "here's the key",
    "the key insight", "worth noting", "crucially", "fundamentally",
    "intuitively", "the part that sounds",
]


def blank_code(text):
    """Remove fenced code blocks but keep line numbers stable."""
    return re.sub(r"```.*?```", lambda m: "\n" * m.group(0).count("\n"),
                  text, flags=re.S)


def kramdown_id(header):
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", header)
    text = re.sub(r"\{:[^}]*\}", "", text)
    text = re.sub(r"^[^a-zA-Z]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9 -]", "", text)
    return text.replace(" ", "-").lower() or "section"


def anchors(md_file):
    ids, seen = set(), {}
    for line in blank_code(md_file.read_text(encoding="utf-8")).splitlines():
        match = HEADER.match(line)
        if match:
            base = kramdown_id(match.group(1))
            count = seen.get(base, 0)
            ids.add(base if count == 0 else f"{base}-{count}")
            seen[base] = count + 1
    return ids


def repo_root(start):
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=start,
                             capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return Path(out.stdout.strip()) if out.returncode == 0 else None


def check(note_path):
    note = Path(note_path).resolve()
    text = note.read_text(encoding="utf-8")
    body = blank_code(text)
    root = repo_root(note.parent)
    problems, warnings, links = [], [], 0

    if not text.startswith("---"):
        problems.append("no front matter at the top of the file")
    if body.count("$$") % 2:
        problems.append("odd number of $$ delimiters (a display formula is not closed)")

    for lineno, line in enumerate(body.splitlines(), start=1):
        for match in IMAGE.finditer(line):
            problems.append(f"line {lineno}: image reference left in the note: {match.group(1)}")

        for match in LINK.finditer(line):
            target = match.group(1)
            links += 1
            blob = GITHUB_BLOB.match(target)
            if blob:
                if root and not (root / unquote(blob.group(1))).exists():
                    problems.append(f"line {lineno}: GitHub link to a file not in the repo: {target}")
                continue
            if re.match(r"^[a-z]+:", target):
                continue
            path, _, anchor = target.partition("#")
            dest = (note.parent / path).resolve() if path else note
            if not dest.exists():
                problems.append(f"line {lineno}: link to a missing file: {target}")
            elif anchor and dest.suffix == ".md" and anchor not in anchors(dest):
                problems.append(f"line {lineno}: link to a missing anchor: {target}")

        lowered = line.lower()
        for phrase in TUTOR_PHRASES:
            if re.search(r"\b" + re.escape(phrase) + r"\b", lowered):
                warnings.append(f'line {lineno}: "{phrase}" sounds like a tutor, reread it')

    return problems, warnings, links


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    problems, warnings, links = check(sys.argv[1])
    for p in problems:
        print(f"FIX    {p}")
    for w in warnings:
        print(f"CHECK  {w}")
    print(f"{links} links checked, {len(problems)} to fix, {len(warnings)} to reread")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
