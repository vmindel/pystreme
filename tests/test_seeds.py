"""Tests for k-mer seed counting/ranking (DESIGNDOC.md §2.1, §5). Per the
project's test-before-optimize convention: `_kmer_ids`/`_revcomp_id` are
checked against plain-Python string manipulation, and the dense (w<=12)
vs sort-based (w>12) counting paths are checked against each other.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import pystreme.seeds as seeds_mod
from pystreme.seeds import _id_to_word, _kmer_ids, _revcomp_id, top_seeds
from pystreme.sequence_store import SequenceStore

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _revcomp_str(word: str) -> str:
    return word.translate(_COMPLEMENT)[::-1]


def _random_seqs(n, length, seed):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list("ACGT"), length)) for _ in range(n)]


def test_top_seeds_binomial_prefilter_keeps_the_exact_top_words():
    # with n_exact smaller than the candidate count, the pre-ranking path
    # must return the same top words (and the same exact p-values) as the
    # all-exact path
    rng = np.random.default_rng(3)
    motif = "GGGGCGGGGG"

    def seqs(n, p):
        out = []
        for _ in range(n):
            b = list(rng.choice(list("ACGT"), 80))
            if rng.random() < p:
                pos = rng.integers(0, 70)
                b[pos : pos + 10] = list(motif)
            out.append("".join(b))
        return out

    primary = SequenceStore.from_sequences(seqs(200, 0.5))
    control = SequenceStore.from_sequences(seqs(200, 0.0))
    exact = top_seeds(primary, control, [8], n_per_width=5, n_exact=10**9)[8]
    fast = top_seeds(primary, control, [8], n_per_width=5, n_exact=50)[8]
    assert [s.word for s in fast] == [s.word for s in exact]
    assert [s.log_pvalue for s in fast] == pytest.approx([s.log_pvalue for s in exact])


def test_kmer_ids_matches_naive_reference():
    seqs = _random_seqs(6, 15, seed=0)
    store = SequenceStore.from_sequences(seqs)
    w = 5
    ids, valid = _kmer_ids(store.codes, store.mask, w)
    for n, seq in enumerate(seqs):
        for i in range(len(seq) - w + 1):
            word = seq[i : i + w]
            expected_id = sum(4 ** (w - 1 - j) * "ACGT".index(c) for j, c in enumerate(word))
            assert int(ids[n, i]) == expected_id
            assert bool(valid[n, i])
            assert _id_to_word(expected_id, w) == word


def test_kmer_ids_masks_windows_spanning_n():
    store = SequenceStore.from_sequences(["ACGTNACGT"])
    _, valid = _kmer_ids(store.codes, store.mask, 3)
    # windows starting at 2,3,4 touch the N at index 4
    expected = [True, True, False, False, False, True, True]
    assert valid[0].tolist() == expected


def test_kmer_ids_rejects_width_longer_than_sequence():
    store = SequenceStore.from_sequences(["ACGT"])
    with pytest.raises(ValueError, match="longer than the sequence length"):
        _kmer_ids(store.codes, store.mask, 10)


@pytest.mark.parametrize("word", ["AAAA", "ACGT", "GGCC", "TTAA", "ACGTA"])
def test_revcomp_id_matches_manual_revcomp(word):
    w = len(word)
    id_ = sum(4 ** (w - 1 - j) * "ACGT".index(c) for j, c in enumerate(word))
    rc_id = int(_revcomp_id(torch.tensor([id_]), w)[0])
    assert _id_to_word(rc_id, w) == _revcomp_str(word)


def test_top_seeds_recovers_implanted_word():
    motif = "GATTACA"
    rng = np.random.default_rng(1)

    def implanted(bg_len, present):
        bases = list(rng.choice(list("ACGT"), bg_len))
        if present:
            pos = rng.integers(0, bg_len - len(motif))
            bases[pos : pos + len(motif)] = list(motif)
        return "".join(bases)

    primary = SequenceStore.from_sequences([implanted(50, True) for _ in range(60)])
    control = SequenceStore.from_sequences([implanted(50, False) for _ in range(60)])

    result = top_seeds(primary, control, widths=[7], n_per_width=3)
    words = {s.word for s in result[7]}
    assert motif in words or _revcomp_str(motif) in words


def test_top_seeds_pools_reverse_complement_counts():
    word = "ACGTAC"
    rc = _revcomp_str(word)
    # primary: 5 literal copies of word, 3 literal copies of its RC, in
    # otherwise-random filler -- top_seeds must report their counts pooled.
    rng = np.random.default_rng(2)

    def build(n_fwd, n_rc):
        bases = list(rng.choice(list("ACGT"), 300))
        s = "".join(bases)
        # splice in non-overlapping literal copies at fixed, spaced offsets
        offsets = list(range(0, 300 - len(word), 20))
        chosen = rng.choice(offsets, n_fwd + n_rc, replace=False)
        s = list(s)
        for i, off in enumerate(chosen):
            s[off : off + len(word)] = list(word if i < n_fwd else rc)
        return "".join(s)

    primary = SequenceStore.from_sequences([build(5, 3)])
    control = SequenceStore.from_sequences(["".join(rng.choice(list("ACGT"), 300))])

    result = top_seeds(primary, control, widths=[len(word)], n_per_width=5)
    seed = next(s for s in result[len(word)] if s.word in (word, rc))
    assert seed.primary_count == 8


def test_top_seeds_dense_and_sort_paths_agree(monkeypatch):
    seqs_p = _random_seqs(15, 40, seed=3)
    seqs_c = _random_seqs(15, 40, seed=4)
    primary = SequenceStore.from_sequences(seqs_p)
    control = SequenceStore.from_sequences(seqs_c)
    w = 5  # 4**5 = 1024, comfortably dense by default

    dense = top_seeds(primary, control, widths=[w], n_per_width=10)[w]

    monkeypatch.setattr(seeds_mod, "_DENSE_COUNT_MAX_W", 1)  # forces the sort-based path for w=5
    sparse = top_seeds(primary, control, widths=[w], n_per_width=10)[w]

    dense_by_word = {s.word: (s.primary_count, s.control_count, round(s.log_pvalue, 6)) for s in dense}
    sparse_by_word = {s.word: (s.primary_count, s.control_count, round(s.log_pvalue, 6)) for s in sparse}
    assert dense_by_word == sparse_by_word


def test_top_seeds_empty_when_no_valid_windows():
    primary = SequenceStore.from_sequences(["N" * 20])
    control = SequenceStore.from_sequences(["ACGT" * 5])
    result = top_seeds(primary, control, widths=[6], n_per_width=3)
    assert result[6] == []


# --- regressions for the 2026-09-08 review --------------------------------


def test_top_seeds_keeps_the_observed_orientation_when_forward_only():
    """Seeds were always canonicalized against their reverse complement,
    even under `revcomp=False`, so forward-only discovery could be handed
    the one orientation its scanner never looks at. `GTTTTC` packs larger
    than `GAAAAC`, so the pooled path reports the RC -- correct only when
    the scan really is double-stranded."""
    word, rc = "GTTTTC", "GAAAAC"
    rng = np.random.default_rng(11)

    def seqs(n, implant):
        out = []
        for _ in range(n):
            b = list(rng.choice(list("ACGT"), 40))
            if implant:
                pos = int(rng.integers(0, 34))
                b[pos : pos + len(word)] = list(word)
            out.append("".join(b))
        return out

    primary = SequenceStore.from_sequences(seqs(200, True))
    control = SequenceStore.from_sequences(seqs(200, False))

    pooled = top_seeds(primary, control, widths=[6], n_per_width=1)[6]
    assert pooled[0].word == rc  # both strands: canonical orientation

    forward = top_seeds(primary, control, widths=[6], n_per_width=1, revcomp=False)[6]
    assert forward[0].word == word  # forward only: the orientation actually present
    assert forward[0].primary_count >= 200
