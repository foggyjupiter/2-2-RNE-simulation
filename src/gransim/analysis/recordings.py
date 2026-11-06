"""Check a folder of new recordings before they are used (docs/RECORDING_PROTOCOL.md).

    gransim check-recordings FOLDER [--rpm 372]

Per file: format (wav, 48 kHz, length), level and clipping, DC offset, and the shaker
speed estimated from the sound — the first-study recordings drifted 6.04-6.64 Hz around a
nominal 6.2 Hz, so a speed far from the one written down is the first thing to catch.
Per group (same particle, N, rpm): level outliers. With recordings.csv: files missing on
either side. Writes FOLDER/check_report.csv and prints a summary; nothing is modified.

File names: ``{particle}_N{count}_rpm{rpm}_{rep}.wav`` (protocol) or the first-study
``{count}-{rep}.wav``; the speed and N come from the name when present.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

PROTOCOL = re.compile(r"^(?P<particle>[^_]+)_N(?P<N>\d+)_rpm(?P<rpm>\d+(?:\.\d+)?)_(?P<rep>\d+)$")
FIRST_STUDY = re.compile(r"^(?P<N>\d+)-(?P<rep>\d+)$")
AUDIO_EXT = {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".3gp"}


def parse_name(stem: str) -> dict:
    m = PROTOCOL.match(stem)
    if m:
        return {"particle": m["particle"], "N": int(m["N"]), "rpm": float(m["rpm"]), "rep": int(m["rep"])}
    m = FIRST_STUDY.match(stem)
    if m:
        return {"particle": "", "N": int(m["N"]), "rpm": None, "rep": int(m["rep"])}
    return {"particle": "", "N": None, "rpm": None, "rep": None}


def check_file(path: Path, nominal_rpm: float) -> dict:
    from scipy.io import wavfile

    from gransim.acoustics.assets import read_wav
    from gransim.analysis.order_params import estimate_shake_frequency

    info = {"file": path.name, **parse_name(path.stem), "issues": []}
    if path.suffix.lower() != ".wav":
        info["issues"].append(f"not a wav ({path.suffix}): convert to 48 kHz wav")
        return info
    try:
        fs, raw = wavfile.read(path)
    except Exception as e:  # unreadable file
        info["issues"].append(f"unreadable: {e}")
        return info
    _, x = read_wav(path)
    dur = len(x) / fs
    rpm_set = info["rpm"] or nominal_rpm
    f_est = estimate_shake_frequency(x, fs, rpm_set / 60.0) if dur >= 2.0 else float("nan")
    info.update({
        "fs": int(fs), "channels": 1 if raw.ndim == 1 else int(raw.shape[1]), "dtype": str(raw.dtype),
        "duration_s": round(dur, 2), "rms": float(np.sqrt(np.mean(x ** 2))),
        "peak": float(np.max(np.abs(x))), "clip_fraction": float(np.mean(np.abs(x) >= 0.99)),
        "dc": float(np.mean(x)), "rpm_set": rpm_set, "rpm_from_sound": round(60.0 * f_est, 1),
    })
    info["rpm_off_pct"] = round(100.0 * (60.0 * f_est / rpm_set - 1.0), 1) if np.isfinite(f_est) else None
    if fs != 48000:
        info["issues"].append(f"sample rate {fs} Hz (protocol: 48 kHz; >24 kHz content matters)")
    if dur < 5.0:
        info["issues"].append(f"only {dur:.1f} s (models use 5 s)")
    info["short"] = 5.0 <= dur < 7.0     # usable, but the protocol asks for ≥ 7 s
    if info["clip_fraction"] > 0.2:
        info["issues"].append(f"{100 * info['clip_fraction']:.0f} % of samples clipped (gain too high?)")
    if info["peak"] < 0.05 and (info["N"] or 0) > 0:
        info["issues"].append("very quiet (peak < 0.05): gain too low or wrong input?")
    if abs(info["dc"]) > 0.02:
        info["issues"].append(f"DC offset {info['dc']:.3f}")
    if info["rpm_off_pct"] is not None and abs(info["rpm_off_pct"]) > 5.0 and (info["N"] or 1) > 0:
        info["issues"].append(f"speed from the sound {info['rpm_from_sound']} rpm is "
                              f"{info['rpm_off_pct']:+.1f} % off the set {rpm_set:g} rpm")
    return info


def check_folder(folder: str | Path, nominal_rpm: float = 372.0, log=print) -> list[dict]:
    folder = Path(folder)
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXT)
    rows = [check_file(p, nominal_rpm) for p in files]
    # level outliers within a group (same particle, N, rpm): > 3 robust SD from the median
    groups: dict = {}
    for r in rows:
        if r.get("rms") is not None and r["N"] is not None:
            groups.setdefault((r["particle"], r["N"], r["rpm_set"]), []).append(r)
    for g in groups.values():
        if len(g) >= 4:
            lv = np.log([max(r["rms"], 1e-6) for r in g])
            med = np.median(lv)
            mad = 1.4826 * np.median(np.abs(lv - med)) + 1e-3
            for r, v in zip(g, lv):
                if abs(v - med) > 3 * mad and abs(v - med) > np.log(1.3):
                    r["issues"].append(f"level {np.exp(v - med):.2f}× the median of its {len(g)} "
                                       "repeats (moved microphone / changed gain?)")
    # recordings.csv cross-check
    sheet = folder / "recordings.csv"
    missing_rows = []
    if sheet.exists():
        with open(sheet, newline="", encoding="utf-8-sig") as fh:
            listed = {(row.get("file") or row.get("파일") or "").strip() for row in csv.DictReader(fh)}
        listed.discard("")
        names = {r["file"] for r in rows}
        missing_rows = sorted(listed - names)
        for r in rows:
            if r["file"] not in listed:
                r["issues"].append("not listed in recordings.csv")
    keys = ["file", "particle", "N", "rpm_set", "rep", "fs", "channels", "duration_s", "rms", "peak",
            "clip_fraction", "dc", "rpm_from_sound", "rpm_off_pct"]
    with open(folder / "check_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(keys + ["issues"])
        for r in rows:
            w.writerow([r.get(k, "") for k in keys] + ["; ".join(r["issues"])])
    bad = [r for r in rows if r["issues"]]
    log(f"{len(rows)} audio files, {len(bad)} with remarks → {folder / 'check_report.csv'}")
    n_short = sum(bool(r.get("short")) for r in rows)
    if n_short:
        log(f"note: {n_short} files are 5-7 s long (usable; the protocol asks for ≥ 7 s so the "
            "centre 5 s avoids the start and stop)")
    sp = [r["rpm_off_pct"] for r in rows if r.get("rpm_off_pct") is not None]
    if sp:
        log(f"speed from the sound vs the set speed: median {np.median(sp):+.1f} %, "
            f"range {min(sp):+.1f} … {max(sp):+.1f} %")
    for r in bad:
        log(f"  {r['file']}: " + "; ".join(r["issues"]))
    for f in missing_rows:
        log(f"  listed in recordings.csv but no file: {f}")
    return rows
