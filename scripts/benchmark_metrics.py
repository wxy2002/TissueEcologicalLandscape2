"""Synthetic performance check, independent of real WSI data.

Run from the repository root with BLAS/OpenMP thread limits set to one.
"""
import argparse
import json
import platform
from pathlib import Path
import resource
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy

from landscape2d import compute_metrics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path)
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
records = []
for size, radius in ((8, float('inf')), (16, float('inf')), (24, float('inf')), (200, 3.0)):
    matrix = np.zeros((size, size), int)
    matrix[::2, ::2] = 1
    timings = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        result = compute_metrics(matrix, search_radius=radius)
        timings.append(time.perf_counter() - start)
    row = dict(size=size, patches=int(result['landscape_patch_count']),
               search_radius='inf' if np.isinf(radius) else radius,
               median_seconds=statistics.median(timings))
    records.append(row)
    print(json.dumps(row), flush=True)
payload = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
               repeats=args.repeats, max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               cases=records)
if args.output:
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
