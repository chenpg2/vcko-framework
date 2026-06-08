# STORY_LOCK — VCKO v3 Secure-Robust-DP Protocol Stack

## The one sub-question this task serves

> **Does the v3 robust-secure-DP protocol stack (geometric-median robust combine + RoFL-style
> norm verification + Turbo-Aggregate secure sum) preserve the v2 principled-aggregation
> predictive utility on the REAL multi-centre IVF data, while adding provable Byzantine
> robustness that the linear combine lacks?**

## Why this sub-question (not scope creep)

The kw-engine knowledge base (41 papers → 71 principles) closed gaps G8/G9/G11 at the
*principle* level. This task validates the G11 protocol stack on real data, not synthetic:
- v2 aggregation (`aggregation.py`, `aggregator.py`) was only validated on a synthetic cohort.
- The Byzantine-robustness claim (median survives a poisoned centre) must hold on real IVF
  coefficient vectors, not just toy data.
- The protocol stack (P-0064 RoFL, P-0065 geometric median, P-0067 Turbo-Aggregate) has zero
  code; this task is its first implementation + real-data evidence.

## Evidence tier

- **Effectiveness** (LOCO AUC on real data) → Correlative/Inferential (held-out centre).
- **Robustness** (geometric median vs linear under a poisoned centre) → Causal (we control
  the corruption, measure the effect).
- **Privacy** (DP utility curve) → Inferential.
- **Protocol correctness** (Turbo-Aggregate secure sum recovers exact aggregate) → Causal/exact.

## Main text vs extended data

- Main: robustness (geometric median vs linear under poisoning) + LOCO AUC parity with v2.
- Extended: DP utility curve, Turbo-Aggregate correctness, norm-verification rejection rate.

## Hard boundaries (what this task does NOT claim)

- We do NOT claim a *formal joint* privacy-utility-robustness rate (that is open gap G16/G11').
- We do NOT implement full cryptographic Bulletproofs ZK proofs; we implement the *norm-bound
  verification logic* (the validity predicate) and document the crypto layer as the production
  substrate. The secure-sum (Turbo-Aggregate masks) IS implemented and tested for exactness.
- Real patient data stays local (`rawinputdata/`, gitignored); no patient-level data is emitted.
