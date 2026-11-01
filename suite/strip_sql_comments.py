#!/usr/bin/env python3
"""Write comment-free copies of the reference SQL for the appendix listings.

The files under `sql-ref/` are what `trql-cli bench --sql` executes, and their
comments carry the argument for why each is a faithful transcription of the
DSL's denotation. The appendix states that argument once in prose, so the
listings it prints are the SQL alone.

Generated, not edited: run this after changing a reference query.

    ./suite/strip_sql_comments.py
"""
from pathlib import Path

QUERIES = ["I24-journey4", "I24-lane-truck", "I24-convoy", "CL-overtake",
           "NS-crossing", "ST-swap", "SD-carpass"]


def cut_trailing(line: str) -> str:
    """Drop a trailing `--` comment, ignoring one that sits inside a literal.

    The reference queries only quote class names, but a scan that tracked no
    quote state would silently truncate a query that did contain `--` in a
    string, and the failure would be a wrong listing rather than an error.
    """
    quoted = False
    i = 0
    while i < len(line):
        c = line[i]
        if c == "'":
            quoted = not quoted
        elif not quoted and c == "-" and line[i + 1:i + 2] == "-":
            return line[:i].rstrip()
        i += 1
    return line.rstrip()


def strip(text: str) -> str:
    """Drop `--` comments, whole-line and trailing, and the blank runs they leave."""
    out: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("--"):
            continue
        line = cut_trailing(line)
        if not line.strip() and (not out or not out[-1].strip()):
            continue
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out) + "\n"


def main() -> None:
    here = Path(__file__).resolve().parent
    src, dst = here / "sql-ref", here.parent / "listings" / "sql"
    dst.mkdir(parents=True, exist_ok=True)
    for q in QUERIES:
        text = (src / f"{q}.sql").read_text()
        stripped = strip(text)
        (dst / f"{q}.sql").write_text(stripped)
        print(f"{q:<16} {len(text.splitlines()):>4} -> {len(stripped.splitlines()):>4} lines")


if __name__ == "__main__":
    main()
