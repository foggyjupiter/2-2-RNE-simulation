"""Command-line entry point.

    gransim run CONFIG.yaml --out runs/demo [--set particles.count=150 --set excitation.frequency=9]
    gransim benchmark [--n 300] [--duration 1.0] [--model linear|hertz]
    gransim info CONFIG.yaml            # print derived dimensionless numbers and warnings

Later stages add: resynthesise audio, sweep, analysis, stiffness convergence,
identifiability check, sim2real comparison.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from gransim.config import SimConfig, check_config, derived_quantities, load_config, save_snapshot


def _parse_sets(items: list[str]) -> dict:
    """Turn ['a.b=1', 'c=x'] into a nested dict with YAML-parsed values."""
    out: dict = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects key=value, got {it!r}")
        key, val = it.split("=", 1)
        d = out
        parts = key.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return out


def cmd_run(args: argparse.Namespace) -> None:
    from gransim.dynamics.engine import run_dynamics

    cfg = load_config(args.config, _parse_sets(args.set))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    check_config(cfg)
    save_snapshot(cfg, out)
    tic = time.perf_counter()
    res = run_dynamics(cfg)
    elapsed = time.perf_counter() - tic
    np.savez_compressed(out / "diagnostics.npz", **res.diagnostics,
                        radius=res.radius, mass=res.mass,
                        final_x=res.final_x, final_v=res.final_v, final_w=res.final_w)
    if res.trajectory is not None:
        np.savez_compressed(out / "trajectory.npz", **res.trajectory)
    from gransim.events.io import save_events, save_wall_force
    from gransim.events.tracker import IMPACT, collision_interval_check

    t0 = res.wall_force.t0
    save_events(out / "events.npz", res.events, res.event_params, (t0, t0 + cfg.time.record_time))
    save_wall_force(out / "wall_force.npz", res.wall_force)
    ev = res.events
    chk = collision_interval_check(ev, res.event_params["binary_contact_time"])
    summary = {"elapsed_s": elapsed, "kernel_s": res.wall_time_s,
               "n_events": int(ev.size),
               "n_impacts": int((ev["contact_type"] == IMPACT).sum()),
               "n_merged_events": int((ev["n_merged"] > 1).sum()),
               "median_impact_interval_s": chk["median_interval"],
               "impact_interval_over_contact_time": chk["ratio"],
               **res.derived}
    if not chk["ok"]:
        print(f"WARNING: median interval between a particle's impacts is only {chk['ratio']:.1f}x "
              "the contact time (spec 4.1 (b)); check with the stiffness convergence test")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_info(args: argparse.Namespace) -> None:
    cfg = load_config(args.config, _parse_sets(args.set))
    print(json.dumps(derived_quantities(cfg), indent=2))
    for w in check_config(cfg):
        print("WARNING:", w)


def cmd_benchmark(args: argparse.Namespace) -> None:
    from gransim.dynamics.engine import Simulation

    contact = {"model": args.model}
    if args.model == "hertz":
        contact["modulus_scale"] = 1.0
    cfg = SimConfig(particles={"count": args.n}, contact=contact,
                    excitation={"frequency": args.frequency}, time={"settle_time": 0.2})
    sim = Simulation(cfg, warn=False)
    sim.step(2)  # JIT compile
    sim.settle()
    tic = time.perf_counter()
    sim.run(args.duration)
    el = time.perf_counter() - tic
    steps = args.duration / sim.dt
    print(f"N={args.n} model={args.model} dt={sim.dt:.3g}s Γ={sim.gamma:.2f}: "
          f"{el:.2f}s for {args.duration}s simulated ({el / steps * 1e6:.1f} µs/step) "
          f"→ 5 s ≈ {5 * el / args.duration:.0f}s")


def _progress(k: int, n: int, r: dict) -> None:
    print(f"[{k}/{n}] N={r['N']} Γ={r['gamma']:.2f} ({r['elapsed_s']:.0f}s)", flush=True)


def cmd_stiffness_test(args: argparse.Namespace) -> None:
    """Spec 10.1 stiffness convergence: run (resumable), evaluate, record allowed stiffness."""
    from gransim.analysis.stiffness import (allowed_levels, build_cases, evaluate, save_allowed,
                                            write_report)
    from gransim.analysis.validation import run_many

    base = load_config(args.config, _parse_sets(args.set)) if args.config else SimConfig(
        time={"settle_time": 0.5, "discard_time": 1.0, "record_time": 2.0})
    conds = [(N, G) for N in args.n for G in args.gamma]
    factors, scales = tuple(args.linear_factors), tuple(args.hertz_scales)
    cases = build_cases(base, conds, seeds=tuple(range(args.seeds)), factors=factors,
                        include_real=True, hertz_scales=scales)
    out = Path(args.out)
    res = run_many([c["config"] for c in cases], workers=args.workers, progress=_progress,
                   cache_dir=out / "cache", time_budget=args.budget)
    if any(r is None for r in res):
        print(f"{sum(r is None for r in res)} cases left; run the same command again to resume")
        return
    summary = evaluate(cases, res, reference="real")
    write_report(summary, out)
    allowed = allowed_levels(summary, base, factors, scales)
    print(json.dumps({"level_pass": summary["level_pass"], "allowed": allowed}, indent=2))
    if args.record:
        save_allowed(allowed, summary, note=f"conditions N={args.n}, Γ={args.gamma}, "
                                            f"seeds={args.seeds}")
        print("recorded allowed stiffness")


def cmd_gamma_c(args: argparse.Namespace) -> None:
    """Spec 10.1 Γc(N): Γ sweep at fixed amplitude, Γc per N, comparison with experiment."""
    from gransim.analysis.gamma_c import (ORDER_PARAMS, comparison_table, extract_gamma_c,
                                          gamma_c_cases, load_reference, trend_increasing)
    from gransim.analysis.validation import run_many

    base = load_config(args.config, _parse_sets(args.set)) if args.config else SimConfig(
        time={"settle_time": 0.5, "discard_time": 1.0, "record_time": 3.0})
    cases = gamma_c_cases(base, args.n, args.gamma, seeds=tuple(range(args.seeds)))
    out = Path(args.out)
    res = run_many(cases, workers=args.workers, progress=_progress, cache_dir=out / "cache",
                   time_budget=args.budget)
    if any(r is None for r in res):
        print(f"{sum(r is None for r in res)} cases left; run the same command again to resume")
        return
    gc = extract_gamma_c(res)
    table = comparison_table(gc, load_reference())
    (out / "sweep_results.json").write_text(json.dumps(res, default=float))
    with open(out / "gamma_c_comparison.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(table[0].keys()) + "\n")
        for row in table:
            fh.write(",".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in row.values()) + "\n")
    for row in table:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()})
    for op in ORDER_PARAMS:
        print(f"{op}: Γc increases with N -> {trend_increasing(gc, op)}")


def cmd_render(args: argparse.Namespace) -> None:
    """Re-render audio of a saved run (no dynamics)."""
    from gransim.acoustics.render import load_run, render_run
    from gransim.config import AcousticConfig, deep_merge

    cfg, *_ = load_run(args.run)
    ac = cfg.acoustics
    sets = _parse_sets(args.set)
    if args.config:
        with open(args.config, encoding="utf-8") as fh:
            sets = deep_merge((yaml.safe_load(fh) or {}).get("acoustics", {}), sets)
    if sets:
        ac = AcousticConfig.model_validate(deep_merge(ac.model_dump(), sets))
    out = Path(args.out or Path(args.run) / "audio.wav")
    meta = render_run(args.run, out, seed=args.seed, acoustics=ac)
    print(json.dumps({k: meta[k] for k in ("duration_s", "rms", "clip_fraction", "n_impacts",
                                            "tau_c_median_s", "noise_file")}, indent=2))
    print(f"wrote {out}")


def cmd_build_assets(args: argparse.Namespace) -> None:
    """Derive the noise library and the response model from first-study recordings."""
    from gransim.acoustics.assets import build_assets

    info = build_assets(args.recordings, args.out, rpm=args.rpm, tau_ref=args.tau_ref,
                        sparse_count=args.sparse_count, noise_count=args.noise_count)
    print(json.dumps(info, indent=2, default=str))


def cmd_sweep(args: argparse.Namespace) -> None:
    """Generate (or resume) a dataset from a sweep spec."""
    from gransim.sweep.runner import run_sweep, write_manifest

    if args.manifest_only:
        print(f"{write_manifest(args.out)} rows")
        return
    print(run_sweep(args.spec, args.out, workers=args.workers, limit=args.limit,
                    time_budget=args.budget, progress=lambda m: print(m, flush=True)))


def cmd_upload(args: argparse.Namespace) -> None:
    """Upload finished runs of a sweep folder to a Hugging Face dataset repo."""
    from gransim.sweep.hf_upload import upload

    extra = {}
    if args.card:
        extra["README.md"] = args.card
    print(f"{upload(args.out, args.repo, extra_files=extra)} files uploaded")


def cmd_identifiability(args: argparse.Namespace) -> None:
    """Spec 8.1: where N and the material mix in the acoustic statistics (no learning)."""
    from gransim.analysis.identifiability import run
    from gransim.sweep.runner import write_manifest

    write_manifest(args.out)
    rep = run(args.out, f_shake=args.f_shake, k=args.k, workers=args.workers)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("by_material", "predictability")},
                     indent=1))
    pr = rep["predictability"]
    print(f"N MAE (linear floor): material unknown {pr['N_mae_material_unknown']:.1f}, "
          f"known {pr['N_mae_material_known']:.1f}, mean guess {pr['N_mean_guess_mae']:.1f}")
    for k, v in pr["properties"].items():
        print(f"  {k:15s} from sound: MAE {v['mae']:.3g} (mean guess {v['mean_guess_mae']:.3g}), R² {v['r2']:.2f}")
    print(f"wrote {Path(args.out) / 'identifiability.json'} and identifiability.png")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="gransim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="single dynamics run")
    p.add_argument("config")
    p.add_argument("--out", required=True)
    p.add_argument("--set", action="append", help="override, e.g. particles.count=150")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("info", help="derived quantities and warnings for a config")
    p.add_argument("config")
    p.add_argument("--set", action="append")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("benchmark", help="time the dynamics kernel")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--frequency", type=float, default=11.0)
    p.add_argument("--model", choices=["linear", "hertz"], default="linear")
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("stiffness-test", help="stiffness convergence test (resumable)")
    p.add_argument("--config")
    p.add_argument("--set", action="append")
    p.add_argument("--out", default="results/stiffness")
    p.add_argument("--n", type=int, nargs="+", default=[150, 300])
    p.add_argument("--gamma", type=float, nargs="+", default=[1.5, 2.3, 6.0])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--linear-factors", type=float, nargs="*", default=[1.0, 4.0, 16.0])
    p.add_argument("--hertz-scales", type=float, nargs="*", default=[1 / 64, 1 / 16, 0.25])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--budget", type=float, default=None, help="stop submitting after N seconds")
    p.add_argument("--record", action="store_true", help="write assets/reference/allowed_stiffness.yaml")
    p.set_defaults(func=cmd_stiffness_test)

    p = sub.add_parser("gamma-c", help="Γc(N) sweep and comparison with experiment (resumable)")
    p.add_argument("--config")
    p.add_argument("--set", action="append")
    p.add_argument("--out", default="results/gamma_c")
    p.add_argument("--n", type=int, nargs="+", default=[100, 150, 200, 250, 300])
    p.add_argument("--gamma", type=float, nargs="+",
                   default=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--budget", type=float, default=None)
    p.set_defaults(func=cmd_gamma_c)

    p = sub.add_parser("render", help="(re)synthesise audio from a saved run")
    p.add_argument("run", help="run folder (events.npz, wall_force.npz, config_snapshot.yaml)")
    p.add_argument("--out", help="output wav (default RUN/audio.wav)")
    p.add_argument("--seed", type=int, default=0, help="acoustic randomness (responses, noise)")
    p.add_argument("--config", help="YAML whose `acoustics:` section overrides the run's")
    p.add_argument("--set", action="append", help="acoustic override, e.g. mic_gain=0.5")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("build-assets", help="noise library + response model from recordings")
    p.add_argument("recordings", help="folder with {N}-{k}.wav clips (first-study layout)")
    p.add_argument("--out", default="assets")
    p.add_argument("--rpm", type=float, required=True, help="shaker rpm of the clips")
    p.add_argument("--tau-ref", type=float, default=5.0e-5,
                   help="real contact time divided out of the click spectrum [s]")
    p.add_argument("--sparse-count", type=int, default=5)
    p.add_argument("--noise-count", type=int, default=0)
    p.set_defaults(func=cmd_build_assets)

    p = sub.add_parser("sweep", help="dataset generation from a sweep spec (resumable)")
    p.add_argument("spec", help="sweep YAML, e.g. configs/sweep_material.yaml")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="max jobs to start in this call")
    p.add_argument("--budget", type=float, default=None, help="stop submitting after N seconds")
    p.add_argument("--manifest-only", action="store_true", help="just rebuild the manifest")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("upload", help="incremental upload of a sweep folder to Hugging Face")
    p.add_argument("out", help="sweep output folder")
    p.add_argument("--repo", required=True, help="dataset repo id, e.g. user/name")
    p.add_argument("--card", help="dataset card (README.md) to upload")
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("identifiability", help="N-vs-material mixing map of a sweep (spec 8.1)")
    p.add_argument("out", help="sweep output folder")
    p.add_argument("--f-shake", type=float, default=6.2)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(func=cmd_identifiability)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
