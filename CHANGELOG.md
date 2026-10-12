# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-11

### Added

1. `SufficiencyEstimator`: a calibrated probability that retrieved passages are enough
   to answer a question, from retrieval, lexical and extractive-reader features
   (logistic regression with isotonic calibration; AUROC, ECE and Brier reported).
2. Selective answering with a guarantee: learn-then-test thresholds (error among
   answered questions at most `alpha` with probability `1 - delta`) and conformal risk
   control (answered-and-wrong rate at most `alpha` in expectation).
3. Recovery actions over any retriever (expand, pseudo-relevance feedback, LLM rewrite,
   hop-by-hop decomposition with `ChatDecomposer`, following names, switching
   retriever) with explicit costs, and `LearnedPolicy`, which
   learns from logged outcomes which action is likely to make the context sufficient
   and trades that off against cost.
4. `Controller`: retrieve, estimate, recover if worthwhile, then answer or abstain, with
   every decision logged.
5. Retrievers: BM25 with an inverted index, dense embeddings and reciprocal-rank hybrid.
6. Generators: an extractive reader, local Hugging Face chat models and any
   OpenAI-compatible endpoint.
7. Benchmarks on SQuAD v2, MuSiQue and HotpotQA: `bench estimate`, `bench selective`
   and `bench recover`, with `scripts/make_figures.py` for every README figure.
8. Command line: `sufficiency score`, `sufficiency ask` and `sufficiency bench`.
9. Shipped estimators for SQuAD v2, MuSiQue and HotpotQA as plain JSON
   (`SufficiencyEstimator.pretrained`).
