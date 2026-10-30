---
name: reviewer
description: Pick the best candidate circuit that passes the equivalence check.
---
Use `quantum_optimizer.reviewer.review(before, candidates)`. Reject any candidate where
`check_equiv` is not ok. Prefer fewer total gates, then fewer CX, then lower depth.
