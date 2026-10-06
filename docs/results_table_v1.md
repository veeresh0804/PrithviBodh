# Results Table v1 (template — fill after Kaggle M1-M5 spatial CV)
All metrics spatial block CV (5-fold GroupKFold, 3-10km) + bootstrap 95% CI. Random-split column only to show inflation.

| ID | Model | OA | macro-F1 [95% CI] | per-class F1 | kappa | random-split macro-F1 | Δ vs best single |
|----|-------|----|-------------------|--------------|-------|----------------------|------------------|
| M1 | RF optical | | | | | | — |
| M2 | RF SAR | | | | | | — |
| M3 | RF/LGBM all early | | | | | | |
| M4 | M1+M2 avg/stack | | | | | | |
| M5 | RF AlphaEarth 64-d | | | | | | |
| B1 | Dynamic World | | | | | — | — |
| B2 | WorldCover | | | | | — | — |

Acceptance: fused − best single Δ CI lower >0 (NFR-01). Cloud monsoon drop fused < optical (NFR-02).
