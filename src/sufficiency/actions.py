"""Recovery actions: ways to get a better context when the current one is not enough.

Each action takes the question and the current passages and returns new passages, plus
what it cost. Costs are counted in calls (one retrieval or one LLM call = 1) so that
policies can trade accuracy against latency and spend.

============  ==================================================================  ======
action        what it does                                                        cost
============  ==================================================================  ======
``keep``      answer with the current passages                                    0
``expand``    retrieve twice as many passages                                     1
``feedback``  add the top passages' most distinctive terms to the query (no LLM)  1
``rewrite``   ask an LLM to rewrite the query, then retrieve                      2
``decompose`` split into sub-questions, retrieve for each in turn, merge          1 + n
``hop``       follow entities named in the top passages to their own passages     3
``switch``    retrieve with a different retriever                                 1
============  ==================================================================  ======

Multi-hop questions need the answer to one hop before the next can be retrieved
("Where was the founder of #1 born?"). ``decompose`` therefore works through the
sub-questions in order and, when an ``answerer`` is given, fills each ``#n`` with the
answer to sub-question ``n`` read from the passages retrieved for it (each answer counts
as a call). ``hop`` needs no decomposition: it retrieves again for the question together
with each of the names that the top passages mention and the question does not.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sufficiency.model import Passage
from sufficiency.retrieval import Retriever, tokenize

Rewriter = Callable[[str], str]
Decomposer = Callable[[str], list[str]]
Answerer = Callable[[str, Sequence[Passage]], str]
_PLACEHOLDER = re.compile(r"#(\d+)")
_NAME = re.compile(r"\b[A-Z][\w'\-]+(?:\s+(?:of|the|de|von|van|[A-Z][\w'\-]+))*")


@dataclass
class Context:
    """Everything an action may use."""

    question: str
    passages: list[Passage]
    retriever: Retriever
    k: int
    alternates: Sequence[Retriever] = ()
    rewriter: Rewriter | None = None
    decomposer: Decomposer | None = None
    answerer: Answerer | None = None
    log: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Outcome:
    passages: list[Passage]
    cost: float
    detail: str


def _merge(*lists: Sequence[Passage], k: int) -> list[Passage]:
    """Interleave ranked lists, dropping duplicates, up to ``k`` passages."""
    seen: set[str] = set()
    out: list[Passage] = []
    longest = max((len(x) for x in lists), default=0)
    for rank in range(longest):
        for ranked in lists:
            if rank < len(ranked) and ranked[rank].id not in seen:
                seen.add(ranked[rank].id)
                out.append(ranked[rank])
                if len(out) >= k:
                    return out
    return out


def keep(ctx: Context) -> Outcome:
    return Outcome(ctx.passages, 0.0, "kept the current passages")


def expand(ctx: Context) -> Outcome:
    k = ctx.k * 2
    return Outcome(ctx.retriever.search(ctx.question, k), 1.0, f"retrieved {k} passages")


def feedback(ctx: Context, terms: int = 5) -> Outcome:
    """Pseudo-relevance feedback: terms frequent in the top passages but absent from the query."""
    query_terms = set(tokenize(ctx.question))
    counts: Counter[str] = Counter()
    for passage in ctx.passages[:3]:
        counts.update(
            t for t in tokenize(f"{passage.title} {passage.text}") if t not in query_terms
        )
    extra = [t for t, _ in counts.most_common(terms)]
    query = f"{ctx.question} {' '.join(extra)}"
    found = ctx.retriever.search(query, ctx.k)
    return Outcome(_merge(found, ctx.passages, k=ctx.k), 1.0, f"expanded the query with {extra}")


def rewrite(ctx: Context) -> Outcome:
    if ctx.rewriter is None:
        return feedback(ctx)
    query = ctx.rewriter(ctx.question)
    found = ctx.retriever.search(query, ctx.k)
    return Outcome(_merge(found, ctx.passages, k=ctx.k), 2.0, f"rewrote the query as {query!r}")


def split_question(question: str) -> list[str]:
    """Heuristic decomposition used when no LLM decomposer is given."""
    parts = re.split(r",? (?:and|but|while) |\? |; ", question)
    parts = [p.strip(" ?") for p in parts if len(p.split()) >= 3]
    return parts if len(parts) > 1 else [question]


def decompose(ctx: Context) -> Outcome:
    subs = (ctx.decomposer or split_question)(ctx.question)
    per = max(2, ctx.k // max(len(subs), 1) + 1)
    found: list[list[Passage]] = []
    answers: list[str] = []
    calls = 1.0 if ctx.decomposer is not None else 0.0
    for sub in subs:
        query = _PLACEHOLDER.sub(
            lambda m: answers[int(m.group(1)) - 1] if int(m.group(1)) <= len(answers) else "",
            sub,
        )
        query = query.replace(">>", " ").strip() or ctx.question
        hits = ctx.retriever.search(query, per)
        found.append(hits)
        calls += 1
        if ctx.answerer is not None and _PLACEHOLDER.search(" ".join(subs)):
            answers.append(ctx.answerer(query, hits).strip())
            calls += 1
        else:
            answers.append("")
    merged = _merge(*found, ctx.passages, k=ctx.k + per)
    detail = f"retrieved for {len(subs)} sub-questions" + (
        f" (bridges: {', '.join(a for a in answers if a)})" if any(answers) else ""
    )
    return Outcome(merged, calls, detail)


def hop(ctx: Context, names: int = 3, per: int = 2) -> Outcome:
    """Retrieve again for the question plus each name the top passages add to it."""
    asked = set(tokenize(ctx.question))
    counts: Counter[str] = Counter()
    for passage in ctx.passages[:3]:
        for name in _NAME.findall(f"{passage.title}. {passage.text}"):
            words = set(tokenize(name))
            if words and not words <= asked:
                counts[name] += 1
    chosen = [n for n, _ in counts.most_common(names)]
    if not chosen:
        return feedback(ctx)
    found = [ctx.retriever.search(f"{ctx.question} {name}", per) for name in chosen]
    merged = _merge(ctx.passages, *found, k=ctx.k + per)
    return Outcome(merged, float(len(chosen)), f"followed {chosen}")


def switch(ctx: Context) -> Outcome:
    if not ctx.alternates:
        return expand(ctx)
    other = ctx.alternates[0]
    found = other.search(ctx.question, ctx.k)
    return Outcome(_merge(found, ctx.passages, k=ctx.k), 1.0, f"retrieved with {other.name}")


ACTIONS: dict[str, Callable[[Context], Outcome]] = {
    "keep": keep,
    "expand": expand,
    "feedback": feedback,
    "rewrite": rewrite,
    "decompose": decompose,
    "hop": hop,
    "switch": switch,
}
