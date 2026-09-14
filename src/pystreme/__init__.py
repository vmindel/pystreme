"""pystreme — GPU-batched reimplementation of STREME for de novo motif discovery.

See DESIGNDOC.md at the repo root for the design and docs/usage.md for how
to use it. The public surface is `MotifDiscovery` (build from a BED +
genome, then `fit`/`scan`/`enrichment`/`discover`), the `Motif` results
it returns, and the table/annotation helpers in `pystreme.results`.
"""

from .discovery import Motif, MotifDiscovery
from .results import annotate, sites_bed, sites_frame, sort_by_pvalue, summary
from .sequence_store import SequenceStore

__version__ = "0.1.0"

__all__ = ["Motif", "MotifDiscovery", "SequenceStore", "annotate", "sites_bed", "sites_frame", "sort_by_pvalue", "summary", "__version__"]
