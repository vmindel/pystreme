"""`fit`, `discover` and `enrichment` must not change `MotifDiscovery.store`.

Each call scores the primary set under a background fitted on its control.
That used to be applied to `disc.store` itself, so `disc.scan(...)` and
`to_meme` gave different answers depending on which calls had run before.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from pystreme import MotifDiscovery
from pystreme.scanner import scan
from pystreme.sequence_store import SequenceStore


def _store(n=120, length=50, seed=0):
    rng = np.random.default_rng(seed)
    site = "TGACTCAT"
    seqs = []
    for i in range(n):
        bases = list(rng.choice(list("ACGT"), length, p=[0.3, 0.2, 0.2, 0.3]))
        if i % 2 == 0:
            pos = int(rng.integers(0, length - len(site)))
            bases[pos : pos + len(site)] = list(site)
        seqs.append("".join(bases))
    store = SequenceStore.from_sequences(seqs)
    store.fit_background(order=1)
    return store


def _snapshot(store):
    return {k: getattr(store, k) for k in ("bg_cumsum", "bg_table", "bg_marginals", "bg_model", "bg_order")}


def _assert_unchanged(store, snap):
    for k, v in snap.items():
        now = getattr(store, k)
        if isinstance(v, torch.Tensor):
            assert now is v, f"store.{k} was replaced"
        else:
            assert now is v or now == v, f"store.{k} changed"


def test_fit_discover_and_enrichment_leave_the_store_background_alone():
    store = _store()
    disc = MotifDiscovery(store, widths=[8], holdout_frac=0.2, seed=0)
    snap = _snapshot(store)
    codes = store.codes.clone()
    pwm = np.full((4, 8), 0.25)
    scores_before = scan([pwm], disc.store, kernel="conv").scores.clone()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kw = dict(n_per_width=2, n_iter=3, kernel="conv")
        disc.enrichment(pwm, kernel="conv", bg_order=2)
        disc.discover(bg_order=2, **kw)
        disc.fit(n_motifs=2, keep_insignificant=True, bg_order=2, **kw)

    _assert_unchanged(store, snap)
    torch.testing.assert_close(store.codes, codes)  # and nothing was erased in it
    torch.testing.assert_close(scan([pwm], disc.store, kernel="conv").scores, scores_before)


def test_to_meme_background_does_not_depend_on_call_history(tmp_path):
    store = _store()
    # a control of different composition: a dinucleotide shuffle keeps the
    # primary's base counts, so its marginals would hide an overwrite
    rng = np.random.default_rng(9)
    control = ["".join(rng.choice(list("ACGT"), store.length, p=[0.1, 0.4, 0.4, 0.1])) for _ in range(store.n_seqs)]
    disc = MotifDiscovery(store, widths=[8], holdout_frac=0.0, seed=0, control=control)
    pwm = {"m": np.full((4, 8), 0.25)}
    disc.to_meme(pwm, tmp_path / "before.meme")
    marginals = disc.store.bg_marginals.clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disc.fit(n_motifs=1, keep_insignificant=True, n_per_width=2, n_iter=3, kernel="conv")
    disc.to_meme(pwm, tmp_path / "after.meme")
    assert (tmp_path / "before.meme").read_text() == (tmp_path / "after.meme").read_text()
    assert torch.equal(disc.store.bg_marginals, marginals)  # exactly, not just to the 4 decimals the file shows


def test_a_bg_order_passed_once_does_not_become_the_default():
    # enrichment(bg_order=0) used to leave store.bg_order at 0, so every
    # later call with the default bg_order=None silently ran order 0
    disc = MotifDiscovery(_store(), widths=[8], holdout_frac=0.0, seed=0)
    disc.enrichment(np.full((4, 6), 0.25), kernel="conv", bg_order=0)
    assert disc.store.bg_order == 1
    primary, control = disc._prepare_primary_and_control(disc.control, None, None, 1, 0)
    assert primary.bg_order == 1 and control.bg_order == 1


def test_fit_results_are_the_same_whatever_ran_before():
    kw = dict(n_motifs=1, keep_insignificant=True, n_per_width=2, n_iter=3, kernel="conv")
    fresh = MotifDiscovery(_store(), widths=[8], holdout_frac=0.2, seed=0)
    used = MotifDiscovery(_store(), widths=[8], holdout_frac=0.2, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        used.enrichment(np.full((4, 6), 0.25), kernel="conv", bg_order=0)
        (a,) = fresh.fit(**kw)
        (b,) = used.fit(**kw)
    np.testing.assert_array_equal(a.pwm, b.pwm)
    torch.testing.assert_close(a.sites.scores, b.sites.scores)
