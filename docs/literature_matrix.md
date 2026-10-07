# Literature Matrix — Batch 58 (M0 exit criterion)

Source: `PROJECT_CONTEXT.md` §2 + §14. URLs below are taken verbatim from §14
(verified via `Select-String -Path PROJECT_CONTEXT.md -Pattern "http"`);
entries with no http URL in the repo cite the plan reference and are marked
"as cited in plan §14, unverified here". Tags [V]/[K] follow the plan.

| # | Paper / product | What it shows (one line) | Number we use (exact metric) | How our plan uses it | Tag |
|---|---|---|---|---|---|
| 1 | Moharrami et al. 2024, RS 16(9):1566 — https://www.mdpi.com/2072-4292/16/9/1566 | S1+S2 beats either sensor alone with SVM/RF | S1+S2 98.25% (migrated samples) vs 87.68% S1, 96.82% S2 — sample-migration accuracy, NOT map accuracy | fusion-benefit prior (M3/M6–M8); caveat quoted in eval | [V] |
| 2 | SAR-optical early vs late fusion, RS 17(7):1298 (2025) — https://www.mdpi.com/2072-4292/17/7/1298/htm | Dual-branch on SEN12MS: early fusion beats late fusion | 88% acc / 87% F1 early vs 84% / 82% late | justifies early/feature fusion configs M6/M7 | [V] |
| 3 | Multi-source S1+S2 time-series DL, ISPRS J. 2019 — https://agritrop.cirad.fr/597770 | Joint S1+S2 time-series DL helps over single modality | Qualitative benefit of both modalities (no single headline number quoted in plan) | fusion-benefit prior; supports 3-season time dim (§6.2) | [V] |
| 4 | Multi-temporal W-Net on S1 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7288459 | Multi-temporal S1 beats single-date S1 | +0.18 acc, +0.25 F1 multi-temporal over single-date | validates seasonal-stack + VH-std feature choice | [V] |
| 5 | Dynamic World (Google/WRI) — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9184477/ | S2-only DL, near-real-time per-image class probabilities | 9 classes, S2-only, per-image probs | baseline B1; sampled at our test points, crosswalked to 6 classes | [V] |
| 6 | Venter et al. 2022, RS 14(16):4101 — as cited in plan §14, unverified here (DOI 10.3390/rs14164101 in repo) | Global-product accuracy is validation-dependent | Global GT: Esri 75%, DW 72%, WC 65%; LUCAS: WC 71%, DW 66%, Esri 63% | sets honest baseline band (~50–75%); no absolute promise upfront | [V] |
| 7 | Malawi comparison — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13407716/ | Same three globals rank differently in Malawi | 65.6% / 46.9% / 61.9% | proves region-dependence; local Hyd validation mandatory | [V] |
| 8 | ESA WorldCover — as cited in plan §14, unverified here | S1+S2 annual map, limited years | 11 classes, annual 2020+2021 only | baseline B2; crosswalked to 6 classes | [V] |
| 9 | Kattenborn et al. 2022, ISPRS Open J — as cited in plan §14, unverified here | Random hold-out inflates CNN accuracy vs spatial blocking | Up to 28% overestimation random vs spatially blocked | validates block-CV choice; report random-split only to show inflation | [V] |
| 10 | TerraMind (ESA Φ-lab + IBM) — https://arxiv.org/abs/2504.11171 + https://research.ibm.com/blog/thinking-in-modalities-terramind + https://sentiwiki.copernicus.eu/web/create-ai-applications | Any-to-any multimodal EO foundation model (S2 L2A, S1 GRD/RTC, DEM, LC, NDVI; in TerraTorch) | Multimodal encoder + Thinking-in-Modalities (generates missing modalities) | main deep model M8, fine-tuned for segmentation | [V] |
| 11 | DOFA — https://arxiv.org/abs/2403.15356 | Wavelength-conditioned multi-sensor FM, S1/S2+, in TorchGeo | Pretrained S1/S2+ backbone | second FM comparison M9 | [V] |
| 12 | Prithvi-EO-2.0 (NASA/IBM) — https://arxiv.org/abs/2412.02732 | Pretrained on HLS optical only | Optical-only FM | optical-only FM baseline (contrasts SAR value-add) | [V] |
| 13 | AlphaEarth Satellite Embedding (Google DeepMind) — https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/ | Ready annual per-pixel embeddings in EE | 64-d per-pixel annual embeddings | strong no-GPU baseline M5 (RF on 64-d) | [V] |
| 14 | SEN12MS dataset — https://arxiv.org/abs/1906.07789 | Large S1+S2 paired dataset with coarse labels | Labels MODIS-derived (coarse) | pretraining data with label-quality caveat | [V] |
| 15 | Dynamic World training set (PANGAEA) — https://doi.pangaea.de/10.1594/PANGAEA.933475 | >5B human-labelled pixels for segmentation pretraining | ~24k tiles 510×510 @10m, expert + crowd + holdout; imagery via EE image IDs, CC BY-4.0 | pretraining data (Stage 1), remapped 9→6 classes | [V] |
| 16 | DW expert-consensus test tiles (Zenodo) — https://zenodo.org/records/4766451 | Independent global consensus test tiles | Global comparison set | external check alongside own Hyd points | [V] |
| 17 | Bhoonidhi (ISRO/NRSC) — https://www.nrsc.gov.in/nrscnew/assets/pdf/brochures/Bhoonidhi_Brochure_2025.pdf | Open Indian EO portal, ≥5 m products incl. LISS-III/IV | Open ≥5 m products | HR reference / qualitative cross-check; extension path | [V] |
| 18 | EOS-04 (ISRO/NRSC) — https://www.nrsc.gov.in/sites/default/files/pdf/EOS_04_writeup_modified.pdf | Indian C-band SAR with terrain-normalised ARD, from Mar 2022 | C-band SAR from Mar 2022 | FR-17 extension to fill 2022–2025 S1B-failure gap (S1B failed 23 Dec 2021; S1C regular from 26 Mar 2025) | [V] |

## What this means for Batch 58

- Fusion is justified but modest: prior gains are +4pp (early-vs-late) to +10pp (S1-only→fused), so NFR-01 tests Δ macro-F1 with bootstrap CI, not absolute accuracy.
- Global products score ~47–75% depending on region/validation, so DW + WorldCover are baselines B1/B2 to beat locally, not ground truth.
- Local labels are mandatory: ~2,000 Hyd test points + ~100 patches, because coarse/global labels (SEN12MS, DW) cannot validate Deccan tanks, granite vs rooftop, or fallow vs bare.
- Block CV is non-negotiable: random splits inflate CNN accuracy up to 28%, so all headlines use spatial blocks with train/test disjointness auto-tested.
- Primary years are 2019 (S1A+B) and 2025 (S1C), because S1B failed Dec 2021 left a coverage gap; EOS-04 covers 2022+ only as an extension.
