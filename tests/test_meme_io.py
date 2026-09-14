"""MEME minimal motif format reader/writer round-trip and parsing tests."""

from __future__ import annotations

import numpy as np
import pytest

from pystreme.meme_io import MemeMotif, read_meme, write_meme


def _random_pwm(width, seed):
    rng = np.random.default_rng(seed)
    raw = rng.random((4, width)) + 0.05
    return raw / raw.sum(axis=0, keepdims=True)


def test_write_read_roundtrip(tmp_path):
    motifs = {
        "MOTIF_A": _random_pwm(8, seed=0),
        "MOTIF_B": _random_pwm(12, seed=1),
    }
    path = tmp_path / "out.meme"
    write_meme(motifs, path)

    parsed = read_meme(path)
    assert set(parsed) == set(motifs)
    for name, pwm in motifs.items():
        np.testing.assert_allclose(parsed[name].pwm, pwm, atol=1e-5)
        assert parsed[name].width == pwm.shape[1]
        # every column of a probability matrix must sum to 1
        np.testing.assert_allclose(parsed[name].pwm.sum(axis=0), 1.0, atol=1e-4)


def test_write_read_roundtrip_with_metadata(tmp_path):
    m = MemeMotif(name="MA0039.1", pwm=_random_pwm(10, seed=2), alt_name="KLF4", nsites=20, evalue=1e-10)
    path = tmp_path / "out.meme"
    write_meme([m], path)

    parsed = read_meme(path)["MA0039.1"]
    np.testing.assert_allclose(parsed.pwm, m.pwm, atol=1e-5)
    assert parsed.nsites == 20
    assert parsed.evalue == pytest.approx(1e-10)


def test_read_meme_minimal_jaspar_style(tmp_path):
    # A trimmed real-world-shaped minimal file, as JASPAR/MEME export it.
    content = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A 0.2500 C 0.2500 G 0.2500 T 0.2500

MOTIF MA0039.1 KLF4
letter-probability matrix: alphabet= ACGT w= 4 nsites= 100 E= 1.2e-050
0.1 0.2 0.3 0.4
0.4 0.3 0.2 0.1
0.25 0.25 0.25 0.25
0.0 0.0 0.5 0.5
"""
    path = tmp_path / "klf4.meme"
    path.write_text(content)

    motifs = read_meme(path)
    assert list(motifs) == ["MA0039.1"]
    m = motifs["MA0039.1"]
    assert m.alt_name == "KLF4"
    assert m.width == 4
    assert m.nsites == 100
    assert m.evalue == pytest.approx(1.2e-50)
    np.testing.assert_allclose(m.pwm[:, 0], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(m.pwm[:, 3], [0.0, 0.0, 0.5, 0.5])
