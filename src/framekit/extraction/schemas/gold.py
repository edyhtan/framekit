"""The benchmark gold set (DESIGN.md §6.2; docs/settings.md §6): the fixed yardstick.

Gold queries with their gold moments, fixed per corpus and independent of any search
space: the harness rejects a search space whose example queries repeat a gold query.

    gold:
      version: 1
      corpus: demo_corpus
      match: {min_overlap: 0.5}       # a hit covers >= 50% of the gold moment
      queries:
        - query: "every fix I push breaks something else"
          moments: [{unit_id: u001, start: 156.5, end: 160.2}]
        - query: "life is hard"
          moments: []                 # no gold moment: the baseline scorer skips it
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field

from framekit.extraction.schemas.common import (
    ConfigError,
    Model,
    Name,
    Tupled,
    UnitId,
    digest,
    parse,
    plain,
    read_mapping,
    unwrap,
)

WHAT = "the gold set"


class GoldMoment(Model):
    unit_id: UnitId
    start: float = Field(ge=0)
    end: float = Field(gt=0)


class GoldQuery(Model):
    query: str = Field(min_length=1)
    moments: Annotated[tuple[GoldMoment, ...], Tupled] = ()


class GoldMatch(Model):
    min_overlap: float = Field(default=0.5, gt=0, le=1)


class GoldSet(Model):
    version: Literal[1] = 1
    corpus: Name
    match: GoldMatch = GoldMatch()
    queries: Annotated[tuple[GoldQuery, ...], Tupled]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GoldSet":
        return parse(cls, plain(unwrap(data, "gold")), WHAT, extra=lambda g: g.problems())

    @classmethod
    def load(cls, path: str | Path) -> "GoldSet":
        return cls.from_mapping(read_mapping(Path(path).expanduser(), WHAT))

    def problems(self) -> list[str]:
        problems = []
        if not self.queries:
            problems.append("queries: at least one query is needed")
        seen: dict[str, int] = {}
        for i, query in enumerate(self.queries):
            key = normalized_query(query.query)
            if key in seen:
                problems.append(f"queries[{i}]: the same query as queries[{seen[key]}]")
            seen.setdefault(key, i)
            for j, moment in enumerate(query.moments):
                if moment.start >= moment.end:
                    problems.append(f"queries[{i}].moments[{j}]: start must be before end")
        return problems

    @property
    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


def normalized_query(text: str) -> str:
    """Lower case, punctuation dropped, spaces collapsed: what counts as the same query."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def overlapping_queries(examples: tuple[str, ...], gold: GoldSet) -> list[str]:
    """Search space examples that repeat a gold query, exactly or nearly (the same words,
    or at least 80% of the words in common). A gold query must never be an example."""
    found = []
    golds = [(q.query, set(normalized_query(q.query).split())) for q in gold.queries]
    for example in examples:
        words = set(normalized_query(example).split())
        for query, gold_words in golds:
            union = words | gold_words
            if union and len(words & gold_words) / len(union) >= 0.8:
                found.append(f"{example!r} repeats the gold query {query!r}")
                break
    return found


def check_gold(gold: GoldSet) -> None:
    problems = gold.problems()
    if problems:
        raise ConfigError(problems, WHAT)


__all__ = [
    "GoldMatch",
    "GoldMoment",
    "GoldQuery",
    "GoldSet",
    "normalized_query",
    "overlapping_queries",
]
