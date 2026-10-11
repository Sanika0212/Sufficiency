"""Answer when the retrieved passages are enough; abstain when they are not.

uv run --extra models python examples/quickstart.py
"""

from sufficiency import BM25, Controller, Passage, SelectiveRule, SufficiencyEstimator
from sufficiency.generate import Extractive

corpus = [
    Passage(
        "einstein",
        "Albert Einstein was born in Ulm, in the Kingdom of Württemberg, in 1879.",
        "Albert Einstein",
    ),
    Passage("ulm", "Ulm is a city on the river Danube in Baden-Württemberg, Germany.", "Ulm"),
    Passage(
        "curie", "Marie Curie was born in Warsaw in 1867 and moved to Paris in 1891.", "Marie Curie"
    ),
    Passage("paris", "Paris is the capital of France and lies on the Seine.", "Paris"),
]
retriever = BM25()
retriever.index(corpus)

estimator = SufficiencyEstimator.pretrained("hotpotqa")  # shipped, fitted on HotpotQA
rule = SelectiveRule(threshold=0.5, method="fixed", alpha=0.0)  # see SelectiveRule.fit
reader = estimator.featurizer.groups[-1]  # the extractive reader the estimator already uses
controller = Controller(retriever, estimator, rule, Extractive(reader), k=5)

for question in [
    "In what year was Albert Einstein born?",
    "In what year did Marie Curie win her second Nobel Prize?",
]:
    result = controller.answer(question)
    verdict = "abstained" if result.abstained else result.answer
    print(f"{question}\n  sufficiency {result.sufficiency:.2f} -> {verdict}")
