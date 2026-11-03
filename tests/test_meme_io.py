"""MEME minimal motif format reader/writer round-trip and parsing tests."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pystreme.meme_io import MemeMotif, _format_evalue, read_meme, write_meme


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


# --- E-values below the smallest double, header robustness ------------------


def _meme_text(header="", alphabet="ALPHABET= ACGT", rows=("0.1 0.2 0.3 0.4", "0.25 0.25 0.25 0.25"), name="m"):
    return (
        f"MEME version 4\n\n{alphabet}\n\nstrands: + -\n\n"
        f"MOTIF {name}\nletter-probability matrix: alength= 4 w= {len(rows)} {header}\n" + "\n".join(rows) + "\n\n"
    )


def test_write_meme_keeps_the_exponent_of_an_evalue_below_the_smallest_double(tmp_path):
    pwm = np.full((4, 3), 0.25)
    log_e = -900.0  # exp(-900) is 0.0 as a double
    path = tmp_path / "m.meme"
    write_meme([MemeMotif("m", pwm, evalue=math.exp(log_e), log_evalue=log_e)], path)
    header = [line for line in path.read_text().splitlines() if line.startswith("letter")][0]
    assert "E= 1.4e-391" in header  # 10 ** (-900 / ln 10) = 10 ** -390.865 = 1.36e-391
    back = read_meme(path)["m"]
    assert math.isclose(back.log_evalue, log_e, abs_tol=0.05)  # one significant digit of mantissa survives


@pytest.mark.parametrize("log_e, text", [(0.0, "1.0e+000"), (math.log(0.05), "5.0e-002"), (math.log(9.97e-5), "1.0e-004")])
def test_format_evalue(log_e, text):
    assert _format_evalue(log_e) == text


def test_write_meme_without_a_log_evalue_is_unchanged(tmp_path):
    path = tmp_path / "m.meme"
    write_meme([MemeMotif("m", np.full((4, 2), 0.25), evalue=0.003)], path)
    assert "E= 0.003" in path.read_text()


def test_read_meme_accepts_a_float_nsites_and_a_deep_evalue(tmp_path):
    path = tmp_path / "m.meme"
    path.write_text(_meme_text("nsites= 20.000000 E= 2.7e-500"))
    m = read_meme(path)["m"]
    assert m.nsites == 20
    assert m.evalue == 0.0  # as a double it underflows...
    assert math.isclose(m.log_evalue, math.log(2.7) - 500 * math.log(10))  # ...the log does not


def test_read_meme_rejects_a_non_dna_alphabet(tmp_path):
    path = tmp_path / "m.meme"
    path.write_text(_meme_text(alphabet="ALPHABET= ACGU"))
    with pytest.raises(ValueError, match="DNA"):
        read_meme(path)
    path.write_text(_meme_text(alphabet='ALPHABET "Protein" PROTEIN-LIKE'))
    with pytest.raises(ValueError, match="DNA"):
        read_meme(path)
    path.write_text(_meme_text(alphabet='ALPHABET "DNA" DNA-LIKE'))
    assert "m" in read_meme(path)


def test_read_meme_rejects_a_row_that_is_not_four_wide(tmp_path):
    path = tmp_path / "m.meme"
    path.write_text(_meme_text(rows=("0.1 0.2 0.7", "0.25 0.25 0.25 0.25")))
    with pytest.raises(ValueError, match="expected 4"):
        read_meme(path)


def test_read_meme_warns_on_a_repeated_name(tmp_path):
    path = tmp_path / "m.meme"
    path.write_text(_meme_text() + _meme_text().split("\n\n", 3)[3])
    with pytest.warns(UserWarning, match="more than once"):
        motifs = read_meme(path)
    assert list(motifs) == ["m"]
