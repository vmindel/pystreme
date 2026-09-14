"""Phase 0 smoke test: the package installs and imports cleanly.

Real correctness tests (L1-L8 in DESIGNDOC.md §4) land alongside each
component starting in Phase 1 — see test_scanner.py, test_statistics.py, etc.
as they're added.
"""

import pystreme


def test_import():
    assert pystreme.__version__


def test_device_fixture_runs(device):
    # Exists mainly to confirm the device-parametrized fixture in conftest.py
    # collects at least "cpu" and, on a GPU node, "cuda" too.
    import torch

    assert torch.zeros(1, device=device).item() == 0.0
