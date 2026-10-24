"""framekit stays lens-free and corpus-free (CLAUDE.md "Architecture rule").

Import boundaries are checked by `uv run lint-imports`; this test checks vocabulary.
Lens and corpus concepts enter only as data (lens specs in A, the corpus profile in V).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEUTRAL_SOURCES = [ROOT / "src"]

FORBIDDEN = re.compile(
    r"\b(emotions?|dynamics|gags?|schadenfreude|backfire|tom|jerry|spike|cartoons?|eras?)\b",
    re.IGNORECASE,
)

# Code plus anything that can carry prompts or text (templates, data).
SCANNED_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".html", ".js"}


def neutral_files():
    for base in NEUTRAL_SOURCES:
        for path in base.rglob("*"):
            if path.suffix in SCANNED_SUFFIXES and "__pycache__" not in path.parts:
                yield path


def test_the_scan_finds_the_library():
    assert any(path.name == "router.py" for path in neutral_files())


def test_library_code_has_no_lens_or_corpus_terms():
    hits = []
    for path in neutral_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert not hits, "lens/corpus terms in library code:\n" + "\n".join(hits)
