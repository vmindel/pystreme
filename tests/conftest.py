"""Shared test fixtures.

Devices: many machines that run this suite have no GPU (a laptop, CI, a cluster
login node), so the correctness suite must be meaningful on CPU
alone; `device` parametrizes over ["cpu"] plus ["cuda"] when available, and
`gpu`-marked tests are skipped automatically when no CUDA device is present.
"""

from __future__ import annotations

import pytest
import torch

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# --- Synthetic genome shared by test_sequence_store.py and test_discovery.py,
# hand-built so every extraction exclusion reason (§2.2) is exercised ---
#
# chrA (len 200): [0:160) "ACGT" x40 (pure), [160:180) "N" x20, [180:200) "ACGT" x5
# chrB (len 30):  "ACGT" x7 + "AC" (pure, no N) -- short, for edge-of-chromosome cases
CHR_A = "ACGT" * 40 + "N" * 20 + "ACGT" * 5
CHR_B = "ACGT" * 7 + "AC"
assert len(CHR_A) == 200
assert len(CHR_B) == 30

SIZE = 20  # half-window = 10


@pytest.fixture
def genome_fasta(tmp_path):
    path = tmp_path / "genome.fa"
    path.write_text(f">chrA\n{CHR_A}\n>chrB\n{CHR_B}\n")
    return path


@pytest.fixture
def peaks_bed(tmp_path):
    # (chrom, start, end, name) -> new_start = center-10, new_end = center+10
    rows = [
        ("chrA", 45, 55, "peak_ok"),  # center=50 -> [40,60) pure pattern
        ("chrA", 0, 4, "peak_edge"),  # center=2 -> new_start=-8, dropped
        ("chrA", 165, 175, "peak_toomanyN"),  # center=170 -> [160,180) all N
        ("chrA", 146, 156, "peak_someN"),  # center=151 -> [141,161), 1 N at the end
        ("chrZ", 0, 10, "peak_unknown_chrom"),  # chrZ doesn't exist
        ("chrB", 0, 10, "peak_chrB_edge"),  # center=5 -> new_start=-5, dropped
        ("chrB", 10, 20, "peak_chrB_ok"),  # center=15 -> [5,25), pure pattern
    ]
    path = tmp_path / "peaks.bed"
    path.write_text("\n".join(f"{c}\t{s}\t{e}\t{n}" for c, s, e, n in rows) + "\n")
    return path


@pytest.fixture(params=_DEVICES)
def device(request) -> str:
    return request.param


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
