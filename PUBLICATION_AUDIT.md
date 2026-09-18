# Project 005 Publication Audit

Status: **publication-ready after corrections**.

## Verified results
- 120,000 feedback interactions; 8,000 customers; 150 products; 168,000 orders.
- VADER macro F1: 0.4883.
- TF-IDF + Logistic Regression macro F1: 1.000 on deliberately template-based synthetic text; this is documented as a synthetic ceiling effect, not a production-performance claim.
- TF-IDF + NMF topic purity: 0.5384 (best of the three compared methods).
- Aspect sentiment agreement: naive keyword + VADER 46.60%; curated phrase lexicon 99.98% on embedded synthetic ground truth.
- Emerging issue: Double Charge share rose from 0.71% baseline to 7.45% recent 7-day share (+954%, z=9.88).
- Corrected operational priority distribution: Critical 516; High 49,075; Medium 58,664; Low 11,745.

## Critical correction made
The original emerging-issue and priority layers consumed `true_themes`, a privileged synthetic ground-truth field. That made business-facing outputs partly oracle-driven. The code and saved outputs now use `inferred_themes`, derived from observable feedback text via the curated lexicon. Ground truth remains only for offline evaluation.

## Additional publication corrections
- Sentiment confidence is explicitly described as model confidence, not a calibrated probability.
- Unsupported generic real-world F1 range was removed from README.
- The churn field is consistently described as a tier-based risk proxy; there is no realized churn-event table.
- Added requirements, CI workflow, and six publication-integrity tests.
