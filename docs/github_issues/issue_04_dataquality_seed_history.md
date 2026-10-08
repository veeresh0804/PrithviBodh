TITLE: [data-quality] Seed acceptance history: 42 rejected → 3 accepted → 6 canonical

**Finding (historical)**
- Checker gates (distance gap ≤5 km, core points both sides, band span ≥4/5):
  seed 42 REJECT (gap 8.39 km, train core 0); seed 3 ACCEPT (gap 3.33 km,
  core 151/48).
- Earlier acceptances for seeds 42 and pre-fix 3 were VOIDED by a
  PYTHONHASHSEED ordering bug — all attempts logged in `docs/seed_log.md`.

**Supersession note (important)**
- Seed 3 belonged to the band design, which was replaced by the **grid design
  with zoned cell assignment**. The canonical skeleton is now **seed 6**
  (`grid_seed: 6`), checksum `449ad05e…`, acceptance: gap 1.05 km, core 177/123,
  min dist 5.055 km — see `docs/seed_log.md` and the decision-meeting pack.

**Human action**
- Guide sign-off (D5) on the split design before real labelling.
