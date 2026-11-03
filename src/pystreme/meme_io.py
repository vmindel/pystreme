"""MEME minimal motif format: reader and writer.

Reader loads JASPAR/CIS-BP PWMs for validation (L1) and for scanning with
known motifs. Writer (`to_meme`) makes pystreme output consumable by Tomtom,
FIMO, and gimmemotifs (DESIGNDOC.md, Goals §1.4 and Non-goals §1). Phase 1.

Format reference: https://meme-suite.org/meme/doc/meme-format.html
Only the minimal subset pystreme needs is implemented: a background line and
one `letter-probability matrix` per motif, alphabet ACGT.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass

import numpy as np

_ALPHABET = "ACGT"


@dataclass
class MemeMotif:
    name: str
    pwm: np.ndarray  # (4, w) probabilities, rows in ACGT order
    alt_name: str = ""
    nsites: int | None = None
    evalue: float | None = None
    # natural-log E-value; when set it is what gets written, so values far
    # below the smallest double (~1e-308) keep their exponent instead of
    # printing as 0
    log_evalue: float | None = None

    @property
    def width(self) -> int:
        return self.pwm.shape[1]


_NUMBER = re.compile(r"^\s*([+]?\d*\.?\d+)(?:[eE]([-+]?\d+))?\s*$")


def _parse_log_evalue(text: str) -> float | None:
    """Natural log of an E-value string, without going through a double,
    so "2.7e-500" reads as about -1151 rather than log(0)."""
    m = _NUMBER.match(text)
    if not m:
        return None
    mantissa = float(m.group(1))
    if mantissa == 0:
        return float("-inf")
    return math.log(mantissa) + int(m.group(2) or 0) * math.log(10)


def _format_evalue(log_e: float) -> str:
    """An E-value from its natural log, as MEME-style `m.me-x` text."""
    if log_e == float("-inf"):
        return "0"
    l10 = log_e / math.log(10)
    exponent = math.floor(l10)
    mantissa = 10 ** (l10 - exponent)
    if round(mantissa, 1) >= 10:  # 9.96 rounds up to 10.0
        mantissa, exponent = mantissa / 10, exponent + 1
    return f"{mantissa:.1f}e{exponent:+04d}"


def _check_alphabet(line: str) -> None:
    if line.startswith("ALPHABET="):
        letters = line.split("=", 1)[1].strip()
        if letters.upper() != _ALPHABET:
            raise ValueError(f"read_meme: alphabet {letters!r} is not supported; pystreme reads DNA (ACGT) motifs only")
    elif line.startswith("ALPHABET") and "DNA" not in line.upper():
        raise ValueError(f"read_meme: custom alphabet {line!r} is not DNA; pystreme reads DNA (ACGT) motifs only")


def read_meme(path) -> dict[str, MemeMotif]:
    """Parse a MEME minimal motif format file into {name: MemeMotif}.

    Raises on a non-DNA alphabet or a matrix row without exactly four
    entries; warns (keeping the later one) when a motif name repeats.
    """
    motifs: dict[str, MemeMotif] = {}
    with open(path) as fh:
        lines = fh.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("ALPHABET"):
            _check_alphabet(line)
            i += 1
        elif line.startswith("MOTIF"):
            parts = line.split()
            name = parts[1]
            alt_name = parts[2] if len(parts) > 2 else ""
            i += 1
            # advance to the letter-probability matrix header for this motif
            while i < len(lines) and not lines[i].strip().startswith("letter-probability matrix"):
                if lines[i].strip().startswith("MOTIF"):
                    raise ValueError(f"MOTIF {name}: no letter-probability matrix block found")
                i += 1
            header = lines[i].strip()
            width = int(_parse_header_field(header, "w"))
            nsites = _parse_header_field(header, "nsites")
            evalue = _parse_header_field(header, "E")
            i += 1

            rows = []
            while len(rows) < width:
                row_line = lines[i].strip()
                i += 1
                if not row_line:
                    continue
                row = [float(x) for x in row_line.split()]
                if len(row) != 4:
                    raise ValueError(f"MOTIF {name}: matrix row {row_line!r} has {len(row)} entries, expected 4 (ACGT)")
                rows.append(row)
            pwm = np.array(rows, dtype=np.float64).T  # (4, w)
            if name in motifs:
                warnings.warn(f"read_meme: motif name {name!r} appears more than once; keeping the later one", stacklevel=2)
            motifs[name] = MemeMotif(
                name=name,
                pwm=pwm,
                alt_name=alt_name,
                nsites=int(round(float(nsites))) if nsites is not None else None,  # some tools write "nsites= 20.000000"
                evalue=float(evalue) if evalue is not None else None,
                log_evalue=_parse_log_evalue(evalue) if evalue is not None else None,
            )
        else:
            i += 1
    return motifs


def _parse_header_field(header: str, field_name: str) -> str | None:
    tokens = header.replace("=", " ").split()
    for j, tok in enumerate(tokens):
        if tok == field_name and j + 1 < len(tokens):
            return tokens[j + 1]
    return None


def write_meme(motifs: dict[str, np.ndarray] | list[MemeMotif], path, background: np.ndarray | None = None):
    """Write motifs to MEME minimal motif format.

    `motifs` is either {name: (4, w) probability array} or a list of
    MemeMotif. `background` is a length-4 ACGT frequency array; uniform by
    default.
    """
    if background is None:
        background = np.full(4, 0.25)

    if isinstance(motifs, dict):
        items = [MemeMotif(name=name, pwm=np.asarray(pwm)) for name, pwm in motifs.items()]
    else:
        items = motifs

    with open(path, "w") as fh:
        fh.write("MEME version 4\n\n")
        fh.write(f"ALPHABET= {_ALPHABET}\n\n")
        fh.write("strands: + -\n\n")
        fh.write("Background letter frequencies\n")
        fh.write(" ".join(f"{b} {p:.4f}" for b, p in zip(_ALPHABET, background)) + "\n\n")

        for m in items:
            header_bits = [f"w= {m.width}"]
            if m.nsites is not None:
                header_bits.append(f"nsites= {m.nsites}")
            if m.log_evalue is not None:
                evalue = _format_evalue(m.log_evalue)
            else:
                evalue = m.evalue if m.evalue is not None else 0
            header_bits.append(f"E= {evalue}")
            fh.write(f"MOTIF {m.name}" + (f" {m.alt_name}" if m.alt_name else "") + "\n")
            fh.write(f"letter-probability matrix: alphabet= {_ALPHABET} " + " ".join(header_bits) + "\n")
            for col in range(m.width):
                fh.write(" ".join(f"{p:.6f}" for p in m.pwm[:, col]) + "\n")
            fh.write("\n")
