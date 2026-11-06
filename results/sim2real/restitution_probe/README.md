# Restitution and acoustic probes vs the first-study recordings

Each `egrid*.json` entry: one simulation setting (dynamics runs in `runs/e_probe/`,
launched by the `launch*.sh` scripts here; all at 372 rpm, BB 6 mm / 0.12 g), rendered
with drr 6, EQ, 500 Hz high-pass and the given impact radiation spread `sigma`; mic gain
calibrated per setting. `table[N][feature] = {real, sim}` means over the 15 recordings
and the renders. Periodicity (`fold_peak_to_mean`) is folded at each clip's own
estimated shaking frequency.

| file | settings |
|---|---|
| egrid.json | e = 0.8 / 0.9 / 0.95 (both pairs) × σ 0, 1.2 |
| egrid_lowe.json | e = 0.5, 0.65; wall relief 0.3°, 0.7°; Γ = 3 |
| egrid_split.json | e_pp 0.95 with e_pw 0.8 / 0.7; e 0.8 relief 0.3° |
| egrid_visco.json | viscoelastic law e(0.5 m/s) = 0.9 / 0.7; split 0.95 / 0.7 with σ 0, 0.6, 1.2 (energy-normalised) |
| radiation_grid.json | DRR 3/6 × position weighting 0/1/2 × σ 0/0.7/1.2 (e = 0.8; older fold metric) |
| attack_grid.json | DRR 3/6 × direct-sound attack 0-0.4 ms, σ 1.2 (older fold metric) |

Summary in `docs/RESEARCH_NOTES.md` §4; figure `results/figures/fig_restitution_probe.png`.
