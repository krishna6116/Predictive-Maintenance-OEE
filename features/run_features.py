"""Phase 3 command line.

Examples (project root, .venv active):
  python -m features.run_features --fit                 # engine split + scaler from FD001 training data
  python -m features.run_features --build train         # features for every training reading
  python -m features.run_features --build test          # features for the test readings loaded so far
  python -m features.run_features --run-id <run_id>     # incremental: only the readings from one batch
"""
import argparse
import sys

from features.build import SCALER_FILE, SPLIT_FILE, add_rul, fit_scaler, load_json, save_json, split_engines
from features.store import build_all, build_for_run, read_readings
from pipeline.db import get_engine


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="FD001")
    p.add_argument("--fit", action="store_true", help="split training engines and fit the scaler")
    p.add_argument("--build", choices=["train", "test"], help="rebuild features for a whole split")
    p.add_argument("--run-id", help="build features only for one pipeline run")
    a = p.parse_args(argv)
    engine = get_engine()

    if a.fit:
        train = add_rul(read_readings(engine, a.dataset, "train"))
        split = split_engines(train.unit.unique())
        scaler = fit_scaler(train, split["train"])
        save_json(split, SPLIT_FILE)
        save_json(scaler, SCALER_FILE)
        print(f"Split: {len(split['train'])} training engines, {len(split['val'])} validation engines -> {SPLIT_FILE}")
        print(f"Scaler fit on {train.unit.isin(split['train']).sum():,} readings from training engines only -> {SCALER_FILE}")

    scaler = load_json(SCALER_FILE) if (a.build or a.run_id) else None
    if a.build:
        n = build_all(engine, a.dataset, a.build, scaler)
        print(f"Built features for {n:,} {a.dataset} {a.build} readings -> fact_features")
    if a.run_id:
        n = build_for_run(engine, a.run_id, scaler)
        print(f"Built features for {n:,} readings from run {a.run_id} -> fact_features")
    if not (a.fit or a.build or a.run_id):
        p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
