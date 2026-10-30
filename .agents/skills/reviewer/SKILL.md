---
name: reviewer
description: Pick the best candidate circuit that passes the equivalence check.
---
Use `quantum_optimizer.reviewer.review(before, candidates, second_opinion=None, records=None)`. If `second_opinion` (e.g. `nemotron.review_rewrite`) is given, it runs on each rewrite before `check_equiv` and is recorded alongside it; it is advisory only. Reject any candidate where
`check_equiv` is not ok. Prefer fewer total gates, then fewer CX, then lower depth.
