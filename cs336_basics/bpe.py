### THIS VERSION IS CHATGPT GENERATED, to optimize the code a bit
### 1dadceff963aea065bc858e3368c7316a48c15fc is the last commit written (mostly) by me for this file 
import multiprocessing as mp
import os
import heapq
import regex as re
from collections import Counter, defaultdict
from array import array

try:
    # Your scaffold helper; kept for drop-in compatibility
    from .pretokenization_example import find_chunk_boundaries  # type: ignore
except Exception:  # pragma: no cover
    def find_chunk_boundaries(f, num_chunks: int, delimiter: bytes) -> list[int]:
        """
        Fallback if the scaffold helper isn't available.
        WARNING: This may split in the middle of UTF-8 sequences.
        """
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if num_chunks <= 1 or size == 0:
            return [0, size]
        step = size // num_chunks
        bounds = [0]
        for i in range(1, num_chunks):
            bounds.append(i * step)
        bounds.append(size)
        return bounds


# --------------------------------------------------------------------------------------
# Deterministic tie-break: prefer lexicographically GREATER (bytes_1, bytes_2) on ties.
# Implemented via a key where larger bytes => smaller key (so heapq pops it first).
# --------------------------------------------------------------------------------------

_INV_TABLE = bytes.maketrans(bytes(range(256)), bytes(range(255, -1, -1)))


def revlex_key(b: bytes) -> bytes:
    """
    Map b -> key such that:
        (a > b)  <=>  (revlex_key(a) < revlex_key(b))
    using normal bytes comparison.

    The 0x00 interleaving + 0xFF terminator fixes the prefix case.
    """
    inv = b.translate(_INV_TABLE)
    n = len(inv)
    out = bytearray(2 * n + 1)
    out[1 : 2 * n : 2] = inv
    out[2 * n] = 0xFF
    return bytes(out)


# --------------------------------------------------------------------------------------
# Pair packing (reduce dict overhead)
# --------------------------------------------------------------------------------------

# Must be > bit_length(max_token_id). 20 supports up to ~1M tokens.
_SHIFT = 20
_MASK = (1 << _SHIFT) - 1


def _pack_pair(a: int, b: int) -> int:
    return (a << _SHIFT) | b


def _unpack_pair(p: int) -> tuple[int, int]:
    return p >> _SHIFT, p & _MASK


# --------------------------------------------------------------------------------------
# Pretokenization (same behavior as your current version; optimized bytes keys)
# --------------------------------------------------------------------------------------

# GPT-2 regex (byte-level BPE standard)
_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(_PAT)

_SPECIAL_RE_CACHE: dict[tuple[str, ...], re.Pattern] = {}


def _get_special_re(special_tokens: list[str]) -> re.Pattern | None:
    """Compile (and cache) a regex that matches any special token."""
    if not special_tokens:
        return None
    # Longer-first avoids prefix issues if overlapping specials exist.
    key = tuple(sorted(special_tokens, key=len, reverse=True))
    pat = _SPECIAL_RE_CACHE.get(key)
    if pat is None:
        special_pattern = "|".join(re.escape(tok) for tok in key)
        pat = re.compile(special_pattern)
        _SPECIAL_RE_CACHE[key] = pat
    return pat


def _pretokenize_into_counts(text: str, out: dict[bytes, int]) -> None:
    """Update `out` with byte-string token counts from `text`."""
    finditer = _PAT_RE.finditer
    out_get = out.get
    out_set = out.__setitem__
    for m in finditer(text):
        wb = m.group(0).encode("utf-8")
        out_set(wb, out_get(wb, 0) + 1)


def pretokenize_bytes(text: str, special_tokens: list[str]) -> dict[bytes, int]:
    """
    Pretokenize into *byte-string* tokens (bytes keys) for speed.

    Splits around special tokens without building an intermediate documents list.
    """
    word_counts: dict[bytes, int] = {}
    special_re = _get_special_re(special_tokens)

    if special_re is None:
        _pretokenize_into_counts(text, word_counts)
        return word_counts

    last = 0
    for sm in special_re.finditer(text):
        start = sm.start()
        if start > last:
            _pretokenize_into_counts(text[last:start], word_counts)
        last = sm.end()

    if last < len(text):
        _pretokenize_into_counts(text[last:], word_counts)

    return word_counts


def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    """Compatibility wrapper: dict[tuple[int,...], int] like original scaffold."""
    counts_b = pretokenize_bytes(text, special_tokens)
    return {tuple(k): v for k, v in counts_b.items()}


def pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> dict[bytes, int]:
    """
    Worker function: read [start:end] from file and pretokenize it.

    Returns dict[bytes,int] to keep keys compact across process boundaries.
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize_bytes(chunk, special_tokens)


def _default_num_processes() -> int:
    """
    Default parallelism for pretokenization.

    Override with env var BPE_NUM_PROCESSES if desired.
    """
    env = os.environ.get("BPE_NUM_PROCESSES")
    if env:
        try:
            n = int(env)
            if n > 0:
                return n
        except ValueError:
            pass
    # Conservative default; on your cluster you likely want to set BPE_NUM_PROCESSES explicitly.
    return max(1, min(8, mp.cpu_count()))


def parallel_pretokenize_bytes(
    input_path: str,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> dict[bytes, int]:
    """
    Parallel pretokenize returning dict[bytes,int].

    Uses streaming aggregation (imap_unordered) to reduce peak memory in the parent.
    """
    if num_processes is None:
        num_processes = _default_num_processes()

    # Fast path: single-process (also avoids needing find_chunk_boundaries).
    if num_processes <= 1:
        with open(input_path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
        return pretokenize_bytes(text, special_tokens)

    delimiter = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, delimiter)

    work_items = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    total = Counter()
    with mp.Pool(num_processes) as pool:
        for part in pool.imap_unordered(pretokenize_chunk, work_items, chunksize=1):
            total.update(part)
    return dict(total)


def parallel_pretokenize(
    input_path: str,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> dict[tuple[int, ...], int]:
    """Backwards-compatible wrapper returning dict[tuple[int,...],int]."""
    counts_b = parallel_pretokenize_bytes(input_path, special_tokens, num_processes=num_processes)
    return {tuple(k): v for k, v in counts_b.items()}


# --------------------------------------------------------------------------------------
# BPE training (major rewrite): exact pair occurrences + local neighbor updates
# --------------------------------------------------------------------------------------

# Sentinel for "null pointer" in the node linked list. Also used for DEAD nodes in sym[].
_NONE = 0xFFFFFFFF


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
    min_count: int = 1,
    verbose_every: int = 100,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train byte-level BPE with *local neighbor updates*.

    Drop-in replacement:
      - Same signature and return types.
      - Same GPT-2 regex pretokenization (no merges across pretoken boundaries).
      - Same deterministic tie-break: if counts tie, merge the lexicographically greatest pair.

    Core speed idea:
      Maintain exact occurrences per pair (a,b). When merging (a,b)->c at a site,
      only pairs touching that site can change:
          (p,a), (a,b), (b,n)  disappear
          (p,c), (c,n)         appear
      We update only those counts and append only those new occurrences.
    """
    # --- 1) Pretokenize (bytes keys) ---
    word_counts_b: dict[bytes, int] = parallel_pretokenize_bytes(
        input_path,
        special_tokens,
        num_processes=num_processes,
    )

    target_vocab_no_special = vocab_size - len(special_tokens)
    if target_vocab_no_special < 256:
        raise ValueError("vocab_size too small (must be >= 256 + len(special_tokens)).")

    # Safety for packed pair keys
    if target_vocab_no_special - 1 >= (1 << _SHIFT):
        raise ValueError(
            f"vocab_size={vocab_size} exceeds packed-pair capacity; increase _SHIFT."
        )

    # --- 2) Vocab (dense ids) + tie-break keys ---
    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    vocab_key: list[bytes] = [revlex_key(tok) for tok in vocab]
    merges: list[tuple[bytes, bytes]] = []

    # --- 3) Build word graphs + initial pair stats ---
    # Word representation:
    #   global node arrays sym/nxt/prv (linked list per word)
    #   head[wid] is the first node of word wid
    #   wcount[wid] is the word frequency
    sym = array("I")
    nxt = array("I")
    prv = array("I")
    head = array("I")
    wcount = array("Q")

    # Pair stats:
    #   pair_counts[pk] = total weighted count
    #   pair_occ[pk] = array('Q') of packed occurrences (wid<<32 | start_node)
    pair_counts: defaultdict[int, int] = defaultdict(int)
    pair_occ: dict[int, array] = {}

    pack_pair = _pack_pair
    pair_occ_get = pair_occ.get

    for wb, c in word_counts_b.items():
        if c < min_count:
            continue

        wid = len(head)
        start = len(sym)
        L = len(wb)

        if L == 0:
            head.append(_NONE)
            wcount.append(0)
            continue

        sym.extend(wb)  # bytes -> ints

        # prev pointers: [_NONE, start, start+1, ...]
        prv.append(_NONE)
        if L > 1:
            prv.extend(range(start, start + L - 1))

        # next pointers: [start+1, start+2, ..., _NONE]
        if L > 1:
            nxt.extend(range(start + 1, start + L))
        nxt.append(_NONE)

        head.append(start)
        wcount.append(c)

        # initial pairs + occurrences
        if L > 1:
            for off in range(L - 1):
                a = wb[off]
                b = wb[off + 1]
                pk = pack_pair(a, b)
                pair_counts[pk] += c
                arr = pair_occ_get(pk)
                if arr is None:
                    arr = array("Q")
                    pair_occ[pk] = arr
                arr.append((wid << 32) | (start + off))

    del word_counts_b  # reduce peak RAM

    # --- 4) Heap with lazy deletion (tie-break matches your old code) ---
    heap: list[tuple[int, bytes, bytes, int]] = []
    entry_finder: dict[int, tuple[int, bytes, bytes, int]] = {}

    for pk, c in pair_counts.items():
        if c <= 0:
            continue
        a, b = _unpack_pair(pk)
        entry = (-c, vocab_key[a], vocab_key[b], pk)
        heap.append(entry)
        entry_finder[pk] = entry
    heapq.heapify(heap)

    heappush = heapq.heappush
    heappop = heapq.heappop
    heapify = heapq.heapify

    def heap_set(pk: int, c: int) -> None:
        a, b = _unpack_pair(pk)
        entry = (-c, vocab_key[a], vocab_key[b], pk)
        entry_finder[pk] = entry
        heappush(heap, entry)

    def heap_pop_best() -> int:
        while heap:
            entry = heappop(heap)
            pk = entry[3]
            if entry_finder.get(pk) is entry:
                del entry_finder[pk]
                return pk
        raise ValueError("No pairs left to merge (corpus too small for requested vocab_size).")

    def maybe_rebuild_heap() -> None:
        if len(heap) > 4 * len(entry_finder) + 200_000:
            heap[:] = list(entry_finder.values())
            heapify(heap)

    # --- 5) Merge loop: local neighbor updates only ---
    sym_a = sym
    nxt_a = nxt
    prv_a = prv
    head_a = head
    wcount_a = wcount
    pair_counts_a = pair_counts
    pair_occ_a = pair_occ

    touched: set[int] = set()
    touched_add = touched.add

    def add_occ(pk: int, wid: int, node: int) -> None:
        arr = pair_occ_a.get(pk)
        if arr is None:
            arr = array("Q")
            pair_occ_a[pk] = arr
        arr.append((wid << 32) | node)

    def merge_at(wid: int, i: int, best_pk: int, a_id: int, b_id: int, new_id: int) -> None:
        wc = wcount_a[wid]
        j = nxt_a[i]         # node with b_id
        p = prv_a[i]         # prev node
        n = nxt_a[j]         # next node after j

        # Remove (p,a)
        if p != _NONE:
            pk = pack_pair(sym_a[p], a_id)
            if pk != best_pk:
                pair_counts_a[pk] -= wc
                touched_add(pk)

        # Remove (b,n) (starts at j)
        if n != _NONE:
            pk = pack_pair(b_id, sym_a[n])
            if pk != best_pk:
                pair_counts_a[pk] -= wc
                touched_add(pk)

        # Apply merge: i becomes new_id, delete j
        sym_a[i] = new_id
        nxt_a[i] = n
        if n != _NONE:
            prv_a[n] = i

        sym_a[j] = _NONE
        nxt_a[j] = _NONE
        prv_a[j] = _NONE

        # Add (p,new) (starts at p)
        if p != _NONE:
            pk = pack_pair(sym_a[p], new_id)
            pair_counts_a[pk] += wc
            touched_add(pk)
            add_occ(pk, wid, p)

        # Add (new,n) (starts at i)
        if n != _NONE:
            pk = pack_pair(new_id, sym_a[n])
            pair_counts_a[pk] += wc
            touched_add(pk)
            add_occ(pk, wid, i)

    # Self-pair merge requires left-to-right to avoid overlap (a==b).
    word_seen: bytearray | None = None
    affected_words: list[int] = []

    total_merges = target_vocab_no_special - 256
    for merge_i in range(total_merges):
        if verbose_every and (merge_i + 1) % verbose_every == 0:
            print(f"Doing merge {merge_i + 1} out of {total_merges}")

        best_pk = heap_pop_best()

        occs = pair_occ_a.pop(best_pk, None)
        if not occs:
            # stale heap entry
            pair_counts_a.pop(best_pk, None)
            maybe_rebuild_heap()
            continue

        a_id, b_id = _unpack_pair(best_pk)

        # New token
        new_id = len(vocab)
        new_bytes = vocab[a_id] + vocab[b_id]
        vocab.append(new_bytes)
        vocab_key.append(revlex_key(new_bytes))
        merges.append((vocab[a_id], vocab[b_id]))

        touched.clear()

        # best_pk will never be merged again; drop its count entry (and we skip updating it)
        pair_counts_a.pop(best_pk, None)

        if a_id == b_id:
            if word_seen is None:
                word_seen = bytearray(len(head_a))

            affected_words.clear()
            for occ in occs:
                wid = occ >> 32
                if not word_seen[wid]:
                    word_seen[wid] = 1
                    affected_words.append(wid)

            for wid in affected_words:
                i = head_a[wid]
                if i == _NONE:
                    continue
                while True:
                    j = nxt_a[i]
                    if j == _NONE:
                        break
                    if sym_a[i] == a_id and sym_a[j] == b_id:
                        merge_at(wid, i, best_pk, a_id, b_id, new_id)
                        i = nxt_a[i]
                        if i == _NONE:
                            break
                    else:
                        i = j

            for wid in affected_words:
                word_seen[wid] = 0

        else:
            for occ in occs:
                wid = occ >> 32
                i = occ & 0xFFFFFFFF

                if sym_a[i] != a_id:
                    continue
                j = nxt_a[i]
                if j == _NONE or sym_a[j] != b_id:
                    continue
                merge_at(wid, i, best_pk, a_id, b_id, new_id)

        # Heap updates for touched pairs only
        for pk in touched:
            c = pair_counts_a[pk]
            if c <= 0:
                pair_counts_a.pop(pk, None)
                entry_finder.pop(pk, None)
                pair_occ_a.pop(pk, None)
            else:
                heap_set(pk, c)

        maybe_rebuild_heap()

    # Add special tokens at the end
    for st in special_tokens:
        vocab.append(st.encode("utf-8"))

    return {i: tok for i, tok in enumerate(vocab)}, merges

def main(
    input_file: str,
    vocab_size: int,
    vocab_output: str,
    merges_output: str,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
    min_count: int = 1,
):
    import json
    import tracemalloc

    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    tracemalloc.start()

    vocab, merges = train_bpe(
        input_file,
        vocab_size,
        special_tokens,
        num_processes=num_processes,
        min_count=min_count,
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory usage (tracemalloc): {peak / 1024 / 1024:.2f} MB")

    longest_token = max(vocab.values(), key=len)
    print(f"Longest token ({len(longest_token)} bytes): {longest_token}")

    vocab_serializable = {str(k): v.hex() for k, v in vocab.items()}
    with open(vocab_output, "w") as f:
        json.dump(vocab_serializable, f, indent=2)

    with open(merges_output, "w") as f:
        for b1, b2 in merges:
            f.write(f"{b1.hex()} {b2.hex()}\n")

    print(f"Saved vocab ({len(vocab)} tokens) to {vocab_output}")
    print(f"Saved merges ({len(merges)} merges) to {merges_output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a byte-level BPE tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the training data file",
    )
    parser.add_argument(
        "vocab_size",
        type=int,
        help="Target vocabulary size (must be >= 256 + number of special tokens)",
    )
    parser.add_argument(
        "--vocab-output",
        type=str,
        default="vocab.json",
        help="Output path for the vocabulary JSON file",
    )
    parser.add_argument(
        "--merges-output",
        type=str,
        default="merges.txt",
        help="Output path for the merges file",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to add to the vocabulary",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes for parallel pretokenization (default: auto)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Minimum word count to include in training (use 2+ to save RAM)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling and print stats at the end",
    )

    args = parser.parse_args()

    if args.profile:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            main(
                input_file=args.input_file,
                vocab_size=args.vocab_size,
                vocab_output=args.vocab_output,
                merges_output=args.merges_output,
                special_tokens=args.special_tokens,
                num_processes=args.num_processes,
                min_count=args.min_count,
            )

        stats = pstats.Stats(pr)
        stats.sort_stats("tottime").print_stats()
    else:
        main(
            input_file=args.input_file,
            vocab_size=args.vocab_size,
            vocab_output=args.vocab_output,
            merges_output=args.merges_output,
            special_tokens=args.special_tokens,
            num_processes=args.num_processes,
            min_count=args.min_count,
        )
