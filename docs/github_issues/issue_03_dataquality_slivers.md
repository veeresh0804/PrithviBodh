TITLE: [data-quality] Sliver cells in grid design (fixed)

**Finding**
- `build_cells` uses ceil, producing a 42 m northern sliver row that was
  assignable for points — a cell so thin it cannot represent its class mix.

**Fix applied**
- Cells below `min_cell_area_fraction` (0.5, from `configs/labels/labelling.yaml`)
  are never assigned or sampled; a test fails if any assigned cell is below it.
- Seed re-accepted after the fix (then superseded by the zoned design — see
  `docs/seed_log.md`).
