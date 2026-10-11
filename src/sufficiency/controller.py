"""The controller: retrieve, judge sufficiency, recover if worthwhile, answer or abstain.

.. code-block:: python

    controller = Controller(retriever, estimator, rule, generator, policy=policy)
    result = controller.answer("Which river flows through the city where Einstein was born?")
    print(result.answer if not result.abstained else "not enough information")
    for step in result.steps:
        print(step.kind, step.detail, step.sufficiency, step.cost)
"""

from __future__ import annotations

from collections.abc import Sequence

from sufficiency.actions import ACTIONS, Context, Decomposer, Rewriter
from sufficiency.estimator import SufficiencyEstimator
from sufficiency.generate import Generator
from sufficiency.model import Result, Step
from sufficiency.policy import LearnedPolicy
from sufficiency.retrieval import Retriever
from sufficiency.selective import SelectiveRule


class Controller:
    def __init__(
        self,
        retriever: Retriever,
        estimator: SufficiencyEstimator,
        rule: SelectiveRule,
        generator: Generator,
        *,
        policy: LearnedPolicy | None = None,
        k: int = 5,
        max_rounds: int = 1,
        alternates: Sequence[Retriever] = (),
        rewriter: Rewriter | None = None,
        decomposer: Decomposer | None = None,
    ) -> None:
        self.retriever = retriever
        self.estimator = estimator
        self.rule = rule
        self.generator = generator
        self.policy = policy
        self.k = k
        self.max_rounds = max_rounds
        self.alternates = alternates
        self.rewriter = rewriter
        self.decomposer = decomposer

    def answer(self, question: str) -> Result:
        passages = self.retriever.search(question, self.k)
        score = self.estimator.predict(question, passages)
        steps = [
            Step("retrieve", f"{len(passages)} passages from {self.retriever.name}", score, 1.0)
        ]
        for _ in range(self.max_rounds):
            if self.rule.answers(score) or self.policy is None:
                break
            decision = self.policy.decide(self.estimator.featurizer(question, passages))
            if decision.action in ("abstain", "keep"):
                steps.append(
                    Step("decide", f"no action is likely to help ({decision.success:.2f})", score)
                )
                break
            ctx = Context(
                question,
                passages,
                self.retriever,
                self.k,
                alternates=self.alternates,
                rewriter=self.rewriter,
                decomposer=self.decomposer,
            )
            outcome = ACTIONS[decision.action](ctx)
            passages = outcome.passages
            score = self.estimator.predict(question, passages)
            steps.append(Step(decision.action, outcome.detail, score, outcome.cost))
        if self.rule.answers(score):
            answer = self.generator(question, passages)
            steps.append(Step("answer", f"answered with {self.generator.name}", score, 1.0))
            return Result(question, answer, False, score, passages, steps)
        steps.append(
            Step("abstain", f"sufficiency {score:.2f} below {self.rule.threshold:.2f}", score)
        )
        return Result(question, None, True, score, passages, steps)
