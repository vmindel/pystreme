"""Control sequence construction: one interface, three modes.

See DESIGNDOC.md §2.6. Phase 2. All three modes expose the same
`generate(store, n_per_seq=1) -> SequenceStore` method, so
`MotifDiscovery.enrichment` (and later, the Phase 4 round loop) can treat
`control="dinuc_shuffle"` / `control="gc_matched"` / `control=<anything else>`
uniformly via `resolve_control`.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import torch

from .sequence_store import SequenceStore, _N_CODE


class DinucShuffle:
    """Per-sequence Euler-path (Altschul-Erikson) k-mer-preserving shuffle.

    Preserves the frequency of every `kmer`-length word in each sequence
    exactly (default `kmer=2`, i.e. dinucleotide frequencies) while
    destroying higher-order structure, and never moves an `N` -- each
    maximal run of non-`N` bases is shuffled independently, in place, so
    separator positions (§2.2's masked positions) land exactly where they
    started.

    Built once per `fit`/`enrichment` call, not per refinement iteration,
    but linear in sequences x `n_per_seq` (~1 ms per 200 bp sequence in
    pure Python). `engine="numba"` (the default whenever numba imports)
    runs the same algorithm compiled, ~50x faster; `engine="python"` is
    the reference implementation and the test oracle.

    Algorithm, per sequence run: build the de Bruijn-style multigraph whose
    nodes are `(kmer-1)`-mers and whose edges are the observed `kmer`-mers
    (for `kmer=2` this is exactly Altschul & Erikson's original mononucleotide
    graph / dinucleotide edges). A uniformly random walk from every node to
    the run's final node, loop-erased as it goes (Wilson's algorithm), fixes
    one "last edge" per node; that's exactly the "spanning in-tree rooted at
    the last node" condition that guarantees a random ordering of the
    remaining edges never gets stuck before every edge is used. The run is
    guaranteed to have at least one valid Eulerian path -- the original
    sequence *is* one -- so this always terminates.
    """

    def __init__(self, kmer: int = 2, seed: int | None = None, engine: str = "auto"):
        if kmer < 1:
            raise ValueError("kmer must be >= 1")
        if engine not in ("auto", "numba", "python"):
            raise ValueError("engine must be 'auto', 'numba', or 'python'")
        self.kmer = kmer
        self.seed = seed
        self.engine = engine

    def generate(self, store: SequenceStore, n_per_seq: int = 1) -> SequenceStore:
        rng = random.Random(self.seed)
        codes = store.codes.cpu().numpy()
        n_seqs, length = codes.shape
        available = _numba_shuffle_available()  # checked, not assumed from the request
        if self.engine == "numba" and not available:
            raise ImportError("DinucShuffle(engine='numba') needs numba installed")
        use_numba = available if self.engine == "auto" else self.engine == "numba"
        out_rows: list[np.ndarray] = []
        if use_numba:
            # numba keeps its own RNG state; seed it once per generate() from
            # the same Python generator the pure-Python path draws from, so
            # DinucShuffle(seed=...) stays deterministic either way.
            _numba_seed(rng.getrandbits(32))
            for _ in range(n_per_seq):
                for row in codes:
                    out_rows.append(_shuffle_row_numba(row, self.kmer))
        else:
            for _ in range(n_per_seq):
                for row in codes:
                    out_rows.append(_shuffle_row(row, self.kmer, rng))
        codes_arr = np.stack(out_rows)
        codes_t = torch.from_numpy(codes_arr).to(device=store.codes.device, dtype=torch.uint8)
        mask_t = codes_t != _N_CODE
        intervals = [
            iv.__class__(iv.chrom, iv.start, iv.end, f"{iv.name}_dinucshuf{rep}")
            for rep in range(n_per_seq)
            for iv in store.intervals
        ]
        return SequenceStore(codes_t, mask_t, intervals, excluded=[], device=store.device)


# --- numba path: the same algorithm on integer arrays -----------------------
#
# The pure-Python version below is the reference (and the test oracle); this
# is a transliteration onto packed (k-1)-mer node ids and CSR out-edge
# lists so a 200 bp sequence shuffles in ~20 us instead of ~1 ms. Only
# loaded if numba is importable; `DinucShuffle(engine=...)` picks.

try:  # pragma: no cover - environment-dependent
    import numba as _numba
except ImportError:  # pragma: no cover
    _numba = None


def _numba_shuffle_available() -> bool:
    return _numba is not None


if _numba is not None:

    @_numba.njit(cache=True)
    def _numba_seed(seed: int) -> None:
        np.random.seed(seed)

    @_numba.njit(cache=True)
    def _euler_shuffle_numba(seq: np.ndarray, k: int) -> np.ndarray:
        length = seq.shape[0]
        if length < k + 1:
            return seq.copy()
        n_edges = length - k + 1
        n_nodes = 4 ** (k - 1)
        # node id per position: packed (k-1)-mer starting there
        ids = np.empty(n_edges + 1, np.int64)
        for i in range(n_edges + 1):
            v = 0
            for j in range(k - 1):
                v = v * 4 + seq[i + j]
            ids[i] = v
        # CSR out-edge lists (edge i goes ids[i] -> ids[i+1], emits seq[i+k-1])
        out_deg = np.zeros(n_nodes, np.int64)
        for i in range(n_edges):
            out_deg[ids[i]] += 1
        offs = np.zeros(n_nodes + 1, np.int64)
        for u in range(n_nodes):
            offs[u + 1] = offs[u] + out_deg[u]
        fill = offs[:-1].copy()
        edges = np.empty(n_edges, np.int64)
        for i in range(n_edges):
            u = ids[i]
            edges[fill[u]] = i
            fill[u] += 1
        end_node = ids[n_edges]

        # Wilson's loop-erased random walks -> one reserved "last edge" per node
        in_tree = np.zeros(n_nodes, np.bool_)
        in_tree[end_node] = True
        last_target = np.full(n_nodes, -1, np.int64)
        pos = np.full(n_nodes, -1, np.int64)
        path = np.empty(n_nodes + 1, np.int64)
        for u in range(n_nodes):
            if out_deg[u] == 0 or in_tree[u]:
                continue
            path[0] = u
            pos[u] = 0
            plen = 1
            cur = u
            while not in_tree[cur]:
                r = np.random.randint(0, out_deg[cur])
                nxt = ids[edges[offs[cur] + r] + 1]
                if pos[nxt] >= 0:  # loop: erase back to nxt
                    cut = pos[nxt]
                    for t in range(cut + 1, plen):
                        pos[path[t]] = -1
                    plen = cut + 1
                else:
                    pos[nxt] = plen
                    path[plen] = nxt
                    plen += 1
                cur = nxt
            for t in range(plen - 1):
                last_target[path[t]] = path[t + 1]
            for t in range(plen):
                in_tree[path[t]] = True
                pos[path[t]] = -1

        # per node: random edge order, the reserved edge last
        for u in range(n_nodes):
            lo, hi = offs[u], offs[u + 1]
            for t in range(hi - 1, lo, -1):  # Fisher-Yates on edges[lo:hi]
                s = lo + np.random.randint(0, t - lo + 1)
                tmp = edges[t]
                edges[t] = edges[s]
                edges[s] = tmp
            if last_target[u] >= 0:
                for t in range(lo, hi):
                    if ids[edges[t] + 1] == last_target[u]:
                        tmp = edges[t]
                        edges[t] = edges[hi - 1]
                        edges[hi - 1] = tmp
                        break

        # Eulerian traversal
        out = np.empty(length, seq.dtype)
        for j in range(k - 1):
            out[j] = seq[j]
        ptr = offs[:-1].copy()
        cur = ids[0]
        for step in range(n_edges):
            e = edges[ptr[cur]]
            ptr[cur] += 1
            out[k - 1 + step] = seq[e + k - 1]
            cur = ids[e + 1]
        return out

    @_numba.njit(cache=True)
    def _shuffle_row_numba(row: np.ndarray, kmer: int) -> np.ndarray:
        out = row.copy()
        n = row.shape[0]
        i = 0
        while i < n:
            if row[i] == 4:
                i += 1
                continue
            j = i
            while j < n and row[j] != 4:
                j += 1
            out[i:j] = _euler_shuffle_numba(row[i:j].astype(np.int64), kmer).astype(row.dtype)
            i = j
        return out


def _shuffle_row(row: np.ndarray, kmer: int, rng: random.Random) -> np.ndarray:
    """Shuffle one (L,) code row, run-by-run, leaving N (code 4) in place."""
    out = row.copy()
    n = len(row)
    i = 0
    while i < n:
        if row[i] == _N_CODE:
            i += 1
            continue
        j = i
        while j < n and row[j] != _N_CODE:
            j += 1
        out[i:j] = _shuffle_run(row[i:j].tolist(), kmer, rng)
        i = j
    return out


def _shuffle_run(seq: list[int], k: int, rng: random.Random) -> list[int]:
    """Random k-mer-frequency-preserving shuffle of one N-free run."""
    length = len(seq)
    if length < k + 1:
        return seq  # not enough symbols for even one edge; nothing to do

    n_edges = length - k + 1

    def node(i: int) -> tuple[int, ...]:
        return tuple(seq[i : i + k - 1])

    start_node = node(0)
    end_node = node(n_edges)

    # Edges carry their emitted symbol explicitly rather than relying on the
    # target node to encode it: for k>=2 the target node's last element
    # *would* be redundant with the symbol, but for k=1 the node is the
    # empty tuple and carries no symbol at all, so this has to hold for both.
    adj: dict[tuple, list[tuple[tuple, int]]] = {}
    for i in range(n_edges):
        u, v, sym = node(i), node(i + 1), seq[i + k - 1]
        adj.setdefault(u, []).append((v, sym))
        adj.setdefault(v, [])

    # --- Step 1: random "last edge" per node, via loop-erased random walk
    # (Wilson's algorithm) towards end_node. Guarantees: following
    # last_edge_target from any node reaches end_node with no cycle -- the
    # spanning-in-tree condition that lets Step 3 never get stuck.
    last_edge_target: dict[tuple, tuple] = {}
    in_tree = {end_node}
    for u in adj:
        if u in in_tree:
            continue
        path = [u]
        pos = {u: 0}
        cur = u
        while cur not in in_tree:
            targets = adj[cur]
            nxt, _sym = targets[rng.randrange(len(targets))]
            if nxt in pos:
                cut = pos[nxt]
                for removed in path[cut + 1 :]:
                    del pos[removed]
                path = path[: cut + 1]
                cur = nxt
            else:
                pos[nxt] = len(path)
                path.append(nxt)
                cur = nxt
        for a, b in zip(path, path[1:]):
            last_edge_target[a] = b
        in_tree.update(path)

    # --- Step 2: per node, random order of all-but-one outgoing edge, with
    # one edge reserved for last: the edge whose target is
    # last_edge_target[u] (root/end_node reserves nothing -- no risk of it
    # stranding itself, per the BEST-theorem construction this implements).
    order: dict[tuple, list[tuple[tuple, int]]] = {}
    for u, edges in adj.items():
        edges = list(edges)
        if u in last_edge_target:
            target = last_edge_target[u]
            last_pos = next(i for i, (t, _s) in enumerate(edges) if t == target)
            last = edges.pop(last_pos)
            rng.shuffle(edges)
            order[u] = edges + [last]
        else:
            rng.shuffle(edges)
            order[u] = edges

    # --- Step 3: walk the fixed edge order from start_node; guaranteed to
    # consume every edge exactly once (that's what the last-edge in-tree
    # from Step 1 buys us) and land on end_node.
    pointer = {u: 0 for u in adj}
    result = list(start_node)
    cur = start_node
    for _ in range(n_edges):
        nxt, sym = order[cur][pointer[cur]]
        pointer[cur] += 1
        result.append(sym)
        cur = nxt
    return result


class GCMatched:
    """Sample a background pool to match the primary set's GC histogram.

    `pool` is a `SequenceStore` of candidate background regions (build one
    with `SequenceStore.from_bed`/`from_fasta`/`from_sequences` -- typically
    random genomic windows of the same width, blacklist-filtered and
    excluding the peaks themselves; that exclusion is the caller's
    responsibility, not checked here).

    GC fraction is computed over unmasked (non-N) positions only. Both the
    primary set and the pool are binned into `n_bins` equal-width bins over
    `[0, 1]`; for each primary sequence, one pool sequence is drawn from the
    matching bin, without replacement while the bin still has unused pool
    sequences, falling back to sampling with replacement (and warning once)
    if a bin runs out.

    **The GC-rich-motif caveat from §2.6 applies**: GC-matching removes
    exactly the compositional signal that makes a GC-box findable. Expect a
    GC-rich motif to weaken or drop out relative to dinucleotide shuffle --
    that's the control doing its job, not a bug. Run both and compare.
    """

    def __init__(self, pool: SequenceStore, n_bins: int = 20, match_repeats: bool = False):
        if match_repeats:
            raise NotImplementedError(
                "GCMatched(match_repeats=True): no repeat annotation is available yet "
                "(would need e.g. RepeatMasker soft-mask fractions wired through "
                "SequenceStore) -- pass match_repeats=False."
            )
        self.pool = pool
        self.n_bins = n_bins
        self.match_repeats = match_repeats

    def generate(self, store: SequenceStore, n_per_seq: int = 1, seed: int | None = None) -> SequenceStore:
        rng = np.random.default_rng(seed)
        primary_gc = _gc_fraction(store.codes.cpu().numpy(), store.mask.cpu().numpy())
        pool_gc = _gc_fraction(self.pool.codes.cpu().numpy(), self.pool.mask.cpu().numpy())

        edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges[-1] = np.nextafter(edges[-1], edges[-1] + 1)  # include gc==1.0 in the last bin
        primary_bins = np.digitize(primary_gc, edges) - 1
        pool_bins = np.digitize(pool_gc, edges) - 1

        pool_by_bin: dict[int, list[int]] = {}
        for idx, b in enumerate(pool_bins):
            pool_by_bin.setdefault(int(b), []).append(idx)
        for indices in pool_by_bin.values():
            rng.shuffle(indices)
        cursors = {b: 0 for b in pool_by_bin}

        # Two different failures, kept apart: a bin the pool *has* but has
        # run out of (the matched GC is still right, it just repeats a
        # sequence), and a bin the pool has *nothing* in (no GC match is
        # possible at all, and the substitute is drawn from the whole pool
        # -- which is not GC matching, so it gets its own warning with the
        # mismatch it actually produced).
        exhausted_bins: set[int] = set()
        empty_bins: set[int] = set()
        chosen: list[int] = []
        for _ in range(n_per_seq):
            for b in primary_bins:
                b = int(b)
                candidates = pool_by_bin.get(b)
                if not candidates:
                    empty_bins.add(b)
                    fallback_pool = [i for indices in pool_by_bin.values() for i in indices] or list(
                        range(self.pool.n_seqs)
                    )
                    chosen.append(int(rng.choice(fallback_pool)))
                    continue
                c = cursors[b]
                if c >= len(candidates):
                    exhausted_bins.add(b)
                    chosen.append(int(rng.choice(candidates)))
                else:
                    chosen.append(candidates[c])
                    cursors[b] = c + 1

        if exhausted_bins:
            warnings.warn(
                f"GCMatched: pool exhausted for {len(exhausted_bins)} GC bin(s) "
                f"({sorted(exhausted_bins)}); sampled with replacement there (the GC "
                "match still holds). Supply a larger pool for strictly "
                "without-replacement matching.",
                stacklevel=2,
            )
        if empty_bins:
            ranges = ", ".join(f"[{edges[b]:.2f}, {edges[b + 1]:.2f})" for b in sorted(empty_bins))
            n_unmatched = sum(1 for b in primary_bins if int(b) in empty_bins) * n_per_seq
            gap = float(np.abs(primary_gc.mean() - pool_gc[chosen].mean()))
            warnings.warn(
                f"GCMatched: the pool has no sequence at all in {len(empty_bins)} GC bin(s) "
                f"({ranges}), so {n_unmatched} control sequence(s) were drawn from the whole "
                f"pool instead and are NOT GC matched (mean primary GC {primary_gc.mean():.3f} vs "
                f"control {pool_gc[chosen].mean():.3f}, gap {gap:.3f}). Supply a pool that covers "
                "the primary GC range, or expect GC itself to differ between the two sets.",
                stacklevel=2,
            )

        codes_t = self.pool.codes[chosen].clone()
        mask_t = self.pool.mask[chosen].clone()
        intervals = [
            self.pool.intervals[i].__class__(
                self.pool.intervals[i].chrom,
                self.pool.intervals[i].start,
                self.pool.intervals[i].end,
                f"{self.pool.intervals[i].name}_gcmatch{rep}",
            )
            for rep in range(n_per_seq)
            for i in chosen[rep * store.n_seqs : (rep + 1) * store.n_seqs]
        ]
        return SequenceStore(codes_t, mask_t, intervals, excluded=[], device=store.device)


def _gc_fraction(codes: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-row GC fraction over unmasked positions (codes 1=C, 2=G)."""
    is_gc = ((codes == 1) | (codes == 2)) & mask
    n_valid = mask.sum(axis=1)
    n_valid = np.where(n_valid == 0, 1, n_valid)  # avoid /0 for a fully-N row; gc reported as 0
    return is_gc.sum(axis=1) / n_valid


class Explicit:
    """User-supplied control sequences: a `SequenceStore`, `list[str]`, or an
    index array selecting rows of a store (e.g. a held-out subset of the
    same primary population).

    Index arrays are resolved against `universe` when `generate` is given
    one, else against `store`. The public API passes the *whole*
    `MotifDiscovery.store` as the universe, which is what its docstrings
    promise: `enrichment(idx=[0, 1], control=[3])` means row 3 of the
    original store, not row 3 of the two-row primary subset (which would
    raise, or silently select some other row).
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def generate(self, store: SequenceStore, n_per_seq: int = 1, universe: SequenceStore | None = None) -> SequenceStore:
        seqs = self.sequences
        if isinstance(seqs, SequenceStore):
            base = seqs
        elif isinstance(seqs, (list, tuple)) and seqs and isinstance(seqs[0], str):
            base = SequenceStore.from_sequences(list(seqs), device=store.device)
        else:
            base = (store if universe is None else universe).subset(np.asarray(seqs).reshape(-1))
        if n_per_seq == 1:
            return base
        reps = [base] * n_per_seq
        codes_t = torch.cat([r.codes for r in reps], dim=0)
        mask_t = torch.cat([r.mask for r in reps], dim=0)
        intervals = [iv for r in reps for iv in r.intervals]
        return SequenceStore(codes_t, mask_t, intervals, excluded=[], device=store.device)


def resolve_control(
    control, store: SequenceStore, n_per_seq: int = 1, seed: int | None = None, universe: SequenceStore | None = None
) -> SequenceStore:
    """Dispatch the `control=` argument from the public API sketch (§3) to a
    concrete control `SequenceStore`, matching `store`'s sequence count
    (times `n_per_seq`).

    Accepts the string shortcuts (`"dinuc_shuffle"`, `"gc_matched"`), an
    already-constructed `DinucShuffle`/`GCMatched`/`Explicit` instance, or
    anything `Explicit` itself accepts (a `SequenceStore`, `list[str]`, or an
    index array) passed directly.

    `seed` makes the randomized controls reproducible (L7): it seeds the
    `"dinuc_shuffle"` shortcut, a `DinucShuffle` instance constructed without
    its own seed, and `GCMatched`'s sampling. An instance's own explicit seed
    always wins over this one.

    `universe` is the store an `Explicit` *index array* indexes into
    (see `Explicit`); generated controls are always built to match
    `store`, i.e. the primary rows actually being tested. They differ
    exactly when the caller restricted the primary set with `idx`.
    """
    if isinstance(control, str):
        if control == "dinuc_shuffle":
            control = DinucShuffle(seed=seed)
        elif control == "gc_matched":
            raise ValueError(
                "control='gc_matched' needs a background pool; pass "
                "GCMatched(pool=<SequenceStore of candidate regions>) instead of the bare string."
            )
        else:
            raise ValueError(f"unknown control string {control!r}; use 'dinuc_shuffle' or a Control instance")
    elif not isinstance(control, (DinucShuffle, GCMatched, Explicit)):
        # anything else (a SequenceStore, list[str], or index array) is
        # exactly what Explicit itself accepts -- checking with `isinstance`
        # rather than `==` here matters: `control == "dinuc_shuffle"` would
        # raise for a numpy index array (elementwise comparison, ambiguous
        # truth value), not just fail to match.
        control = Explicit(control)
    if isinstance(control, DinucShuffle) and control.seed is None and seed is not None:
        control = DinucShuffle(kmer=control.kmer, seed=seed, engine=control.engine)  # seed it, don't reconfigure it
    if isinstance(control, GCMatched):
        return control.generate(store, n_per_seq=n_per_seq, seed=seed)
    if isinstance(control, Explicit):
        return control.generate(store, n_per_seq=n_per_seq, universe=universe)
    return control.generate(store, n_per_seq=n_per_seq)
