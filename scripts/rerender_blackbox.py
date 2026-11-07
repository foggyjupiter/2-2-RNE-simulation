"""Re-render a downloaded sweep with the black-box acoustics.

    python scripts/rerender_blackbox.py SRC OUT [--workers 4] [--limit N]

SRC holds runs/<id>/{config_snapshot.yaml, events.npz, wall_force.npz, done.json}
(e.g. the v3/ or lowe/ folder of the Hugging Face dataset, without audio). See
src/gransim/acoustics/blackbox.py for the acoustic model and its parameter ranges.
"""
import argparse
import json

from gransim.sweep.rerender_blackbox import rerender_dataset

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
print(json.dumps(rerender_dataset(a.src, a.out, workers=a.workers, limit=a.limit)), flush=True)
