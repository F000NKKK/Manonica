"""Summarize raw Lab timings after a wall-clock warm-up (standard library only)."""
import argparse
import csv
import math
import sys

METRICS = (
    "interval_ms", "render_ms", "input_ms", "update_ms", "overlays_ms",
    "draw_present_ms",
)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace")
    parser.add_argument("--warmup-seconds", type=float, default=10)
    args = parser.parse_args()
    if not math.isfinite(args.warmup_seconds) or args.warmup_seconds < 0:
        parser.error("warmup must be finite and nonnegative")
    elapsed = 0.0
    samples = []
    with open(args.trace, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not set(METRICS).issubset(reader.fieldnames or []):
            raise ValueError("missing timing columns")
        for line, row in enumerate(reader, 2):
            values = {key: float(row[key]) for key in METRICS}
            if any(not math.isfinite(value) or value < 0 for value in values.values()):
                raise ValueError(f"invalid timing on line {line}")
            start = elapsed
            elapsed += values["interval_ms"] / 1000
            if start >= args.warmup_seconds:
                samples.append(values)
    if not samples:
        raise ValueError("no frames remain after warm-up")
    print(f"Frames: {len(samples)}; warm-up: {args.warmup_seconds:g}s")
    print(f"{'metric (ms)':24} {'p50':>10} {'p95':>10} {'max':>10}")
    for key in METRICS:
        values = [row[key] for row in samples]
        print(f"{key:24} {percentile(values, .5):10.3f} "
              f"{percentile(values, .95):10.3f} {max(values):10.3f}")
    # gpu_batch_ms is a retained snapshot, not an independent batch per frame.
    print("GPU batches are retained snapshots; use gpu-coupled for batch statistics.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
