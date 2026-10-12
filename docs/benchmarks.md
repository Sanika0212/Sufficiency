# Benchmarks

Every number in the README comes from these commands. Datasets are downloaded from the
Hugging Face Hub on first use and cached in `~/.cache/sufficiency` (set
`SUFFICIENCY_CACHE` to move it). Runs write `results.json` and `results.md` under
`runs/`.

```bash
uv sync --extra models --extra bench --extra viz
```

## 1. Does the context suffice? (`bench estimate`)

```bash
uv run sufficiency bench estimate --datasets squad_v2 musique hotpotqa --train 1500 --test 1000
```

1. **Data.**
   1. SQuAD v2 (train for fitting, validation for testing): one paragraph per
      question; sufficient means answerable (a third of SQuAD v2 questions are written
      to look answerable from their paragraph but are not).
   2. MuSiQue (answerable subset; train and validation): 20 paragraphs per question,
      2 to 4 of them supporting, one per hop. Sufficient means every supporting
      paragraph is in the top 5 that BM25 returns.
   3. HotpotQA distractor (validation only, so the first 1,500 questions fit and the
      next 1,000 test): 10 paragraphs, 2 supporting; for half the questions the
      supporting paragraphs are removed. Sufficient means both supporting paragraphs
      are in the top 5.
2. **Scorers.** The calibrated estimator; the estimator without the reader features;
   a hand-weighted blend (0.25 top score, 0.20 score drop-off, 0.25 coherence between
   passages, 0.30 question coverage); the reader's span-versus-no-answer margin alone;
   the retrieval score share alone.
3. **Measures.** AUROC; expected calibration error (10 equal-width bins) and Brier
   score for the scorers that output probabilities (the hand-weighted blend is clipped
   to [0, 1] and treated as one, since that is how such scores are used).

## 2. Answer or abstain with a guarantee (`bench selective`)

```bash
uv run sufficiency bench selective --generator extractive --correctness grounded --out runs/selective
uv run sufficiency bench selective --generator local:Qwen/Qwen2.5-1.5B-Instruct \
    --correctness match --out runs/selective-qwen-match
```

Uses the held-out questions and the estimators from benchmark 1. Each question is
answered from its top 5 passages. Two definitions of a correct answer:

1. **grounded**: the context is sufficient and the answer matches a reference
   (contains it, or token F1 at least 0.5). An answer from an insufficient context
   counts as wrong even when it matches, because it could only be a guess.
2. **match**: the answer matches a reference, as in standard selective question
   answering. A model that knows the answer, or reads it from a partial context, gets
   credit.

On SQuAD v2 the two coincide (sufficient means answerable).

The held-out questions are split in half, one half to calibrate each rule and the other
to measure it, and every number is the average over 200 random halvings: a guarantee is
about the rule's true error, and a single test half of 500 questions fluctuates around
it by a few points.

1. **AURC**: area under the risk-coverage curve (error among answered questions against
   the share answered, highest score first). Lower is better.
2. **Learn then test** at targets of 20% and 50% error among answered questions, with
   `delta = 0.1`: the achieved error and coverage. The fixed sequence starts at the
   strictest threshold answering at least 10% of the calibration questions.
3. **Conformal risk control** at a 10% target for the share of all questions answered
   wrongly.

Generators: `extractive` (the SQuAD 2.0 reader's best span; no LLM),
`local:<model>` (a Hugging Face chat model, greedy decoding, 32 new tokens), or
`openai:<model>` (any OpenAI-compatible endpoint; `OPENAI_API_KEY` from the
environment or `.env`). Answers are cached under the output folder.

## 3. Which fix works, at what cost? (`bench recover`)

```bash
uv run sufficiency bench recover --questions 1000 --k 5
```

1. **Setting.** Open retrieval: the paragraphs of 1,000 MuSiQue validation questions
   are pooled into one corpus of about 20,000 paragraphs. Each question starts from
   BM25's top 5.
2. **Actions**, each run once per question with its outcome logged: `keep`, `expand`
   (top 10), `feedback` (pseudo-relevance feedback), `decompose` (heuristic split,
   retrieved hop by hop), `hop` (follow names the top passages mention), `switch`
   (dense retrieval with all-MiniLM-L6-v2), `decompose-llm` (Qwen2.5-1.5B writes the
   sub-questions from three hand-written examples, linking hops with `#1`; retrieved hop
   by hop with the reader filling each `#n`), and `decompose-gold` (MuSiQue's annotated
   sub-questions, hop by hop; an upper bound, not deployable).
3. **Outcome.** Sufficient: every supporting paragraph is in the new context. Also
   reported: the share of supporting paragraphs found, and the cost in calls (a
   retrieval or a model call each count 1) actually made.
4. **Policies**, fitted on the first 500 questions and measured on the other 500: each
   action for every question, a rule cascade on the hand-weighted score
   (keep at 0.7 or more, feedback from 0.4, else expand), the learned policy at cost
   weights 0 to 0.2, and the oracle (the cheapest deployable action that worked, in
   hindsight).

## Figures

```bash
uv run python scripts/make_figures.py
```

Reads `runs/estimate`, `runs/selective` and `runs/recover/results.json` and writes the
README figures to `docs/assets/` in light and dark versions.
