# How Sufficiency works

A retrieval-augmented system answers from the passages its retriever returns. When those
passages do not contain what the question needs, a language model usually answers
anyway, and the answer is a guess dressed as a fact. Sufficiency puts three decisions
between retrieval and generation:

1. **Is the context enough?** A calibrated probability that the passages suffice.
2. **Can it be fixed cheaply?** If not enough, which recovery action is most likely to
   make it enough, given what each action costs.
3. **Answer or abstain?** A threshold on the probability, chosen so that the error rate
   is guaranteed on new questions.

```text
question ──▶ retrieve ──▶ estimate P(sufficient) ──▶ above the threshold? ──yes──▶ answer
                               ▲                          │ no
                               │                          ▼
                               └──── run the action ◀── learned policy: best action
                                                          (or abstain if none is likely to help)
```

## 1. Estimating sufficiency

### The label

A context is *sufficient* when the question can be answered from it. On the benchmarks
this is measured, not judged by a model:

1. SQuAD v2: the question is answerable from its paragraph (the dataset marks
   unanswerable questions written to look answerable).
2. MuSiQue: every supporting paragraph (2 to 4 per question, one per hop) is among the
   top `k` retrieved.
3. HotpotQA: both supporting paragraphs are among the top `k`; for half the questions
   the gold paragraphs are removed, so only distractors remain.

### Features

Three groups, each usable alone (`sufficiency.features`):

1. **Retrieval**: how decisively the retriever ranked the passages: the top score's share
   of the total, the gap to the runner-up, the entropy of the score distribution.
2. **Lexical**: how much of the question's content the passages cover (in the top
   passage and in all of them), the number of question terms and capitalised names, a
   cue that the question has several hops, and the question word.
3. **Reader**: an extractive reader trained with unanswerable questions
   (`deepset/roberta-base-squad2`) scores the best answer span against its "no answer"
   option, in every passage. Features are the largest and mean margin, the best span's
   probability and how often passages agree on the answer.

### The estimator

`SufficiencyEstimator` is logistic regression on standardised features (balanced class
weights), wrapped in isotonic calibration with 5-fold cross-validation. Calibration is
the point: the selection rules below need a score that can be thresholded, and a user
needs "0.8" to mean that about 80 out of 100 such contexts really are enough. Every
benchmark reports expected calibration error (ECE, 10 bins) and the Brier score next
to AUROC.

The common alternative is a fixed blend of signals with hand-set weights (for example
0.25 top score, 0.20 score drop-off, 0.25 coherence between passages, 0.30 question
coverage). `bench estimate` includes exactly such a blend as a baseline.

## 2. Answering with a guarantee

`sufficiency.selective` turns the estimate into a rule "answer if the score is at least
`t`". A labelled calibration set (scores, and whether answering would have been wrong)
picks `t` for one of two promises. Both assume new questions are exchangeable with the
calibration questions.

### Conformal risk control (`crc`)

Promise: the share of *all* questions that are answered *and* wrong is at most `alpha`,
in expectation. With `n` calibration questions and loss `L_i(t) = 1[score_i >= t and
wrong_i]`, which never increases as `t` grows, the rule picks the smallest `t` with

```text
n / (n + 1) * mean_i L_i(t) + 1 / (n + 1) <= alpha
```

which gives `E[L_new(t)] <= alpha` (conformal risk control). It answers as much as the
bound allows.

### Learn then test (`ltt`)

Promise: among *answered* questions the error rate is at most `alpha`, with probability
at least `1 - delta` over the calibration set. This is usually the promise a product
needs ("when it answers, it is right 90% of the time").

Thresholds on a grid are tested from strict to lenient. At each threshold the null
hypothesis is "the error rate among answered questions exceeds `alpha`", tested with
the exact binomial tail `P(X <= errors)`, `X ~ Binomial(answered, alpha)`, at level
`delta`. Testing stops at the first threshold that fails; because the order is fixed in
advance, no multiplicity correction is needed (fixed-sequence testing). The rule uses
the most lenient threshold that passed.

## 3. Recovering cheaply

When the context is not enough, there are several standard fixes, each with a price.
`sufficiency.actions` implements them over any retriever:

| Action | What it does | Cost |
|---|---|---|
| `keep` | nothing | 0 |
| `expand` | retrieve twice as many passages | 1 retrieval |
| `feedback` | add frequent terms from the top passages to the query and retrieve again | 1 retrieval |
| `rewrite` | have an LLM rewrite the query (falls back to `feedback`) | 1 LLM call + 1 retrieval |
| `decompose` | split the question into sub-questions and retrieve for each, hop by hop | 1 LLM call (if used) + 1 retrieval and 1 reader call per hop |
| `hop` | retrieve again with each name the top passages mention and the question does not | 1 retrieval per name |
| `switch` | retrieve with a different retriever (for example dense instead of BM25) | 1 retrieval |

Multi-hop questions need hop-by-hop decomposition: "Where was the founder of #1 born?"
can only be retrieved once hop 1 is answered. `decompose` therefore works through the
sub-questions in order, and a reader fills each `#n` with the answer to hop `n` from
the passages retrieved for it. `ChatDecomposer` asks an LLM for such sub-questions.

Which action helps depends on the question. Expanding helps when the answer is just
below the cut-off; decomposition helps when a later hop's entity never appears in the
question; switching helps with paraphrase. Hand-built pipelines encode this as a fixed
cascade ("if confidence is middling, rewrite; if low, expand").

`LearnedPolicy` learns it instead. On training questions every action is run once and
the outcome logged (1 if the new context is sufficient). One calibrated classifier per
action predicts `P(success | action, features of the original context)`. At run time
the policy chooses

```text
argmax over actions a of  P(success | a, x) - cost_weight * cost(a)
```

and abstains when even the best action's predicted success is below `abstain_below`.
Sweeping `cost_weight` traces the frontier between sufficiency and cost.
`bench recover` compares it with each fixed action, a rule cascade on the hand-weighted
score, and the oracle (the cheapest action that worked, in hindsight).

## The controller

`Controller` runs the loop: retrieve, estimate, and if the rule says abstain and a
policy is configured, run the chosen action and estimate again (up to `max_rounds`
times). It then answers with the generator or abstains. `Result.steps` records every
decision with its sufficiency estimate and cost, so a log of the run shows why the
system answered or why it did not.

## Limitations

1. The guarantees hold for questions like the calibration questions. Calibrate on
   questions from your own traffic; a guarantee from SQuAD does not transfer to a
   support desk.
2. Sufficiency is about the context, not the generator. A sufficient context can still
   be answered wrongly, which is why the selective benchmark counts an answer as
   correct only when it also matches the reference.
3. The reader is trained on SQuAD 2.0, so SQuAD v2 is in-domain for it. MuSiQue and
   HotpotQA test transfer.
4. Recovery outcomes are measured as "the needed passages are now present". This
   isolates retrieval; it does not show that a generator then answers correctly.
