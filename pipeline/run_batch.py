"""Run one pipeline batch: ingest -> validate -> load.

Examples (from the project root, with .venv active):
  python -m pipeline.run_batch --init                         # create tables and sensor limits
  python -m pipeline.run_batch --split train --all            # load the full training set once
  python -m pipeline.run_batch --next                         # load the next 5 test cycles ("one shift")
  python -m pipeline.run_batch --next --inject-faults         # same, with 3 bad rows added to show quarantine
  python -m pipeline.run_batch --start-cycle 1 --cycles 5     # (re)load a specific window; safe to repeat
"""
import argparse
import sys

from pipeline.db import apply_schema, get_engine
from pipeline.ingest import load_file, next_batch
from pipeline.load import last_received_cycles, load_batch, next_start_cycle
from pipeline.validate import compute_limits, inject_faults, load_limits, save_limits
from pipeline.validate import validate as check
from pipeline.config import LIMITS_FILE


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="FD001")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--init", action="store_true", help="create tables and compute sensor limits, then exit")
    p.add_argument("--start-cycle", type=int)
    p.add_argument("--cycles", type=int, default=5, help="cycles per batch (one 'shift')")
    p.add_argument("--next", action="store_true", help="continue after the last successful batch")
    p.add_argument("--all", action="store_true", help="load every cycle in one batch")
    p.add_argument("--inject-faults", action="store_true", help="add 3 bad rows to demonstrate quarantine")
    args = p.parse_args(argv)

    engine = get_engine()
    if args.init or not LIMITS_FILE.exists():
        apply_schema(engine)
        save_limits(compute_limits(load_file(args.dataset, "train")))
        print(f"Tables ready; sensor limits written to {LIMITS_FILE}.")
        if args.init:
            return 0

    df = load_file(args.dataset, args.split)
    if args.all:
        start, n = 1, int(df.cycle.max())
    else:
        start = next_start_cycle(engine, args.dataset, args.split) if args.next else (args.start_cycle or 1)
        n = args.cycles

    batch = next_batch(df, start, n)
    if batch.empty:
        print(f"No readings at cycles {start}-{start + n - 1}: every engine has stopped reporting.")
        return 0
    if args.inject_faults:
        batch = inject_faults(batch)

    good, bad = check(batch, load_limits(), last_received_cycles(engine, args.dataset, args.split))
    run_id = load_batch(engine, good, bad, dataset=args.dataset, split=args.split, start_cycle=start,
                        n_cycles=n, note="fault injection demo" if args.inject_faults else None)

    print(f"Run {run_id}: {args.dataset} {args.split}, cycles {start}-{start + n - 1}, "
          f"{batch.unit.nunique()} engines -> {len(good)} loaded, {len(bad)} quarantined")
    for why, count in bad.reason.value_counts().items():
        print(f"  quarantined: {count} x {why}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
