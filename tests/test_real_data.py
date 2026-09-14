"""Real-data smoke test: mm10 genome + a real SP1 ChIP peak set.

Needs a real genome and an SP/KLF ChIP peak set (the checks look for the SP1
motif), given by the PYSTREME_TEST_GENOME and PYSTREME_TEST_PEAKS environment
variables; without them the file skips itself. This is a
lightweight sanity check that extraction/background-fitting don't fall over
on real, messy genomic data; the actual scanner-vs-FIMO comparison (L1) lands
once the scanner exists.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from pystreme.meme_io import read_meme
from pystreme.scanner import scan
from pystreme.sequence_store import SequenceStore

GENOME = os.environ.get("PYSTREME_TEST_GENOME", "")  # genome FASTA with .fai
PEAKS = os.environ.get("PYSTREME_TEST_PEAKS", "")  # summit-centred SP/KLF ChIP peaks
FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(GENOME) and os.path.exists(PEAKS)),
    reason="real genome/peak paths not available on this machine",
)


def _subset_bed(tmp_path, n_lines):
    subset = tmp_path / "subset.bed"
    with open(PEAKS) as fh, open(subset, "w") as out:
        for i, line in enumerate(fh):
            if i >= n_lines:
                break
            out.write(line)
    return subset


def test_extract_sp1_peaks_subset(tmp_path):
    subset = _subset_bed(tmp_path, 200)
    store = SequenceStore.from_bed(subset, GENOME, size=200)

    # a handful may legitimately be dropped (chrom edge / N content); most shouldn't be
    assert store.n_seqs > 190
    assert store.length == 200
    assert store.codes.dtype == torch.uint8
    # no fully-N sequences slipped through the max_n_frac filter
    assert (store.mask.float().mean(dim=1) > 0.8).all()


def test_background_fit_on_real_data(tmp_path):
    subset = _subset_bed(tmp_path, 500)
    store = SequenceStore.from_bed(subset, GENOME, size=200)
    store.fit_background(order=2)

    assert store.bg_cumsum.shape == (store.n_seqs, store.length + 1)
    assert torch.isfinite(store.bg_table).all()
    probs = store.bg_table.exp()
    torch.testing.assert_close(probs.sum(dim=1), torch.ones(probs.shape[0]), atol=1e-4, rtol=0)


def test_sp1_motif_scores_above_background_on_real_sp1_peaks(tmp_path):
    """First real-motif-on-real-data check: does the scanner actually find
    the ChIP'd factor's own motif in its own peaks?

    This is deliberately not the rigorous L1 (scanner-vs-FIMO) or L6
    (STREME head-to-head) validation from DESIGNDOC.md -- just a sanity
    signal, using the real JASPAR SP1 motif against the real SP1 peak set
    supplied for this project. What it does check, and what held when this
    test was written: SP1 scores above the order-2 background on the large
    majority of peaks, and its best-site positions are somewhat more central
    than KLF4's (a similar but not-the-ChIP'd-factor GC-box binder) -- in the
    right direction, though only modestly above chance. Real central
    enrichment statistics are Phase 4 (§2.7) work; this test only asserts
    the robust part.
    """
    subset = _subset_bed(tmp_path, 1000)
    store = SequenceStore.from_bed(subset, GENOME, size=200)
    store.fit_background(order=2)

    sp1 = read_meme(os.path.join(FIXTURES, "MA0079.1_SP1.meme"))["MA0079.1"]
    klf4 = read_meme(os.path.join(FIXTURES, "MA0039.1_KLF4.meme"))["MA0039.1"]
    result = scan([sp1.pwm, klf4.pwm], store, revcomp=True, kernel="gather")

    sp1_scores, klf4_scores = result.scores[0], result.scores[1]
    assert (sp1_scores > 0).float().mean().item() > 0.85

    center = (store.length - sp1.width) / 2
    sp1_offset = result.positions[0].float() - center
    klf4_offset = result.positions[1].float() - center
    sp1_central_frac = (sp1_offset.abs() < 25).float().mean().item()
    klf4_central_frac = (klf4_offset.abs() < 25).float().mean().item()
    assert sp1_central_frac > klf4_central_frac


def test_fit_finds_a_gc_box_de_novo_in_real_sp1_peaks(tmp_path):
    """Phase 4's `fit` on real data: with no motif given, the top de novo
    motif in SP1 ChIP peaks should be a GC-box (SP1's own GGGGCGGGG /
    CCCCGCCCC family), significant on held-out peaks and centrally
    enriched. Kept small (1000 peaks, two widths) so it stays a quick
    check -- the full-size STREME head-to-head (L6) is
    scripts/validate_streme.py, run via bsub, not pytest.
    """
    from pystreme.discovery import MotifDiscovery

    subset = _subset_bed(tmp_path, 1000)
    disc = MotifDiscovery.from_bed(subset, GENOME, size=200, bg_order=2, seed=0)
    motifs = disc.fit(n_motifs=1, widths=[8, 10], n_per_width=3, n_iter=10, kernel="gather")

    assert len(motifs) == 1
    m = motifs[0]
    consensus = m.consensus
    rc = consensus.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]
    assert "GGGCGG" in consensus or "GGGCGG" in rc or "GGGGCG" in consensus or "GGGGCG" in rc, m
    assert m.holdout_logp < np.log(1e-3)
    assert m.central_logp < np.log(1e-3)
    assert m.n_sites > 300
