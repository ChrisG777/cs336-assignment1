### THIS VERSION IS CHATGPT GENERATED, to optimize the code a bit
### 1dadceff963aea065bc858e3368c7316a48c15fc is the last commit written (mostly) by me for this file 
import multiprocessing as mp
import os
import heapq
import regex as re
from collections import Counter, defaultdict
from array import array

from .pretokenization_example import find_chunk_boundaries

# --------------------------------------------------------------------------------------
# Ordering helper for deterministic tie-breaks
# --------------------------------------------------------------------------------------

_INV_TABLE = bytes.maketrans(bytes(range(256)), bytes(range(255, -1, -1)))


def revlex_key(b: bytes) -> bytes:
    """
    Key with the property:
        a > b   <=>   revlex_key(a) < revlex_key(b)
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
# Pair packing (reduce dict/set overhead)
# --------------------------------------------------------------------------------------

# Must be > bit_length(max_token_id). 20 supports up to ~1M tokens.
_SHIFT = 20
_MASK = (1 << _SHIFT) - 1


def _pack_pair(a: int, b: int) -> int:
    return (a << _SHIFT) | b


def _unpack_pair(p: int) -> tuple[int, int]:
    return p >> _SHIFT, p & _MASK


# --------------------------------------------------------------------------------------
# Pretokenization (optimized: bytes keys, streaming aggregation)
# --------------------------------------------------------------------------------------

# GPT-2 regex (as required by the assignment handout)
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
    """
    Compatibility wrapper: returns dict[tuple[int,...], int] like the original scaffold.
    """
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
    A memory-friendlier default than mp.cpu_count().

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
    # On large files, spawning N=cpu_count() workers can blow up RAM because each worker
    # builds a big dict and also pickles it back to the parent.
    return max(1, min(4, mp.cpu_count()))


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

    # find_chunk_boundaries expects a delimiter bytes pattern.
    delimiter = special_tokens[0].encode("utf-8") if special_tokens else b"\n"

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, delimiter)

    work_items = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    total = Counter()

    # Fast path for 1 process (saves mp overhead + avoids pickling large dicts).
    if num_processes == 1:
        for wi in work_items:
            total.update(pretokenize_chunk(wi))
        return dict(total)

    with mp.Pool(num_processes) as pool:
        for part in pool.imap_unordered(pretokenize_chunk, work_items, chunksize=1):
            total.update(part)

    return dict(total)


def parallel_pretokenize(
    input_path: str,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> dict[tuple[int, ...], int]:
    """
    Backwards-compatible wrapper returning dict[tuple[int,...],int].
    """
    counts_b = parallel_pretokenize_bytes(input_path, special_tokens, num_processes=num_processes)
    return {tuple(k): v for k, v in counts_b.items()}


# --------------------------------------------------------------------------------------
# Word encoding helpers (major memory win vs tuple[int,...])
# --------------------------------------------------------------------------------------

def _choose_word_format(target_vocab_no_special: int) -> tuple[str, str, int]:
    """
    Choose a compact packed representation for word token-id sequences.

    Returns (array_typecode, memoryview_format, itemsize_bytes).

    - 'H' (uint16) is enough up to 65535 token ids.
    - 'I' (uint32) supports larger vocab sizes (still < 2^32).
    """
    if target_vocab_no_special <= 0xFFFF:
        return "H", "H", 2
    if target_vocab_no_special <= 0xFFFFFFFF:
        return "I", "I", 4
    raise ValueError("vocab_size too large for packed word encoding")


def _pack_word_from_bytes(wb: bytes, typecode: str) -> bytes:
    """
    Convert a byte string (sequence of initial token ids 0..255) into
    packed uint16/uint32 bytes so we can store merged token ids compactly.
    """
    # IMPORTANT: array(typecode, wb) treats wb as *raw binary*, requiring len(wb)
    # to be a multiple of itemsize. We want element-wise widening instead.
    arr = array(typecode)
    arr.extend(wb)  # bytes iterates as ints 0..255
    return arr.tobytes()


def _transform_word_packed(
    word: bytes,
    mv_fmt: str,
    a: int,
    b: int,
    new_token: int,
    out_typecode: str,
) -> tuple[bytes, bool]:
    """
    Replace occurrences of (a,b) left-to-right in a packed word.

    Returns (new_word, changed_flag).
    """
    mv = memoryview(word).cast(mv_fmt)
    n = len(mv)
    if n < 2:
        return word, False

    out = array(out_typecode)
    append = out.append
    i = 0
    changed = False

    while i < n:
        if i + 1 < n and mv[i] == a and mv[i + 1] == b:
            append(new_token)
            i += 2
            changed = True
        else:
            append(mv[i])
            i += 1

    if not changed:
        return word, False
    return out.tobytes(), True


# --------------------------------------------------------------------------------------
# BPE training
# --------------------------------------------------------------------------------------

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
    min_count: int = 1,
    verbose_every: int = 100,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train byte-level BPE.

    Args:
      input_path: path to a UTF-8-ish text file
      vocab_size: total vocab size INCLUDING special tokens
      special_tokens: list of specials to append at the end
      num_processes: worker processes for pretokenization (lower -> less RAM)
      min_count: drop pre-tokens with corpus frequency < min_count (saves lots of RAM; set 1 for exact)
      verbose_every: print progress every N merges (0 disables)

    Returns:
      vocab: dict[token_id -> token_bytes]
      merges: list[(bytes_1, bytes_2)] in merge order
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
        raise ValueError(f"vocab_size={vocab_size} exceeds packed-pair capacity; increase _SHIFT.")

    word_typecode, mv_fmt, itemsize = _choose_word_format(target_vocab_no_special)

    # --- 2) Convert pretokenized bytes -> packed word bytes in ONE pass (lower peak RAM) ---
    words: list[bytes | None] = []
    counts = array("Q")  # uint64 counts
    word_to_id: dict[bytes, int] = {}

    for wb, c in word_counts_b.items():
        if c < min_count:
            continue
        w = _pack_word_from_bytes(wb, word_typecode)
        wid = len(words)
        words.append(w)
        counts.append(c)
        word_to_id[w] = wid

    # free the big pretoken dict
    del word_counts_b

    # --- 3) Vocab + tie-break keys (dense ids) ---
    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    vocab_key: list[bytes] = [revlex_key(tok) for tok in vocab]
    merges: list[tuple[bytes, bytes]] = []

    # --- 4) pair_counts and pair_to_words ---
    pair_counts: dict[int, int] = defaultdict(int)  # pk -> total frequency
    # pk -> array('I') of word ids that contain pk at least once.
    # NOTE: we never remove individual ids; deactivated words are skipped by counts[wid]==0.
    pair_to_words: dict[int, array] = {}

    for wid, w in enumerate(words):
        c = counts[wid]
        if c == 0 or w is None or len(w) < 2 * itemsize:
            continue

        mv = memoryview(w).cast(mv_fmt)
        n = len(mv)
        if n < 2:
            continue

        prev = mv[0]
        seen_pairs: set[int] = set()
        for i in range(1, n):
            tok = mv[i]
            pk = _pack_pair(prev, tok)
            pair_counts[pk] += c

            if pk not in seen_pairs:
                arr = pair_to_words.get(pk)
                if arr is None:
                    arr = array("I")
                    pair_to_words[pk] = arr
                arr.append(wid)
                seen_pairs.add(pk)

            prev = tok

    # --- 5) Heap with entry_finder + periodic rebuild ---
    heap: list[tuple[int, bytes, bytes, int]] = []
    entry_finder: dict[int, tuple[int, bytes, bytes, int]] = {}

    heappush = heapq.heappush
    heappop = heapq.heappop
    heapify = heapq.heapify

    def heap_set(pk: int, c: int) -> None:
        """Set/update the heap priority for pk to count c (must be >0)."""
        a, b = _unpack_pair(pk)
        entry = (-c, vocab_key[a], vocab_key[b], pk)
        entry_finder[pk] = entry
        heappush(heap, entry)

    for pk, c in pair_counts.items():
        if c > 0:
            heap_set(pk, c)

    def heap_pop_best() -> tuple[int, int]:
        """Return (pair_key, count) for the current max pair."""
        while heap:
            entry = heappop(heap)
            pk = entry[3]
            if entry_finder.get(pk) is entry:
                del entry_finder[pk]
                return pk, -entry[0]
        raise ValueError("No pairs left to merge (corpus too small for requested vocab_size).")

    def maybe_rebuild_heap() -> None:
        # If heap is mostly stale, rebuild from entry_finder values.
        if len(heap) > 4 * len(entry_finder) + 200_000:
            heap[:] = list(entry_finder.values())
            heapify(heap)

    # --- 6) Helpers to update pair_counts + pair_to_words for new words ---
    def adjust_pair_counts(seq: bytes | None, delta: int, touched: set[int], skip_pk: int) -> None:
        if seq is None or len(seq) < 2 * itemsize:
            return
        mv = memoryview(seq).cast(mv_fmt)
        n = len(mv)
        if n < 2:
            return
        prev = mv[0]
        for i in range(1, n):
            tok = mv[i]
            pk = _pack_pair(prev, tok)
            if pk != skip_pk:
                pair_counts[pk] += delta
                touched.add(pk)
            prev = tok

    def add_word_to_pairs(wid: int, seq: bytes, skip_pk: int) -> None:
        """Add wid to pair_to_words for each unique pair in seq."""
        if len(seq) < 2 * itemsize:
            return
        mv = memoryview(seq).cast(mv_fmt)
        n = len(mv)
        if n < 2:
            return
        prev = mv[0]
        seen: set[int] = set()
        for i in range(1, n):
            tok = mv[i]
            pk = _pack_pair(prev, tok)
            if pk != skip_pk and pk not in seen:
                arr = pair_to_words.get(pk)
                if arr is None:
                    arr = array("I")
                    pair_to_words[pk] = arr
                arr.append(wid)
                seen.add(pk)
            prev = tok

    # --- 7) Merge loop ---
    total_merges = target_vocab_no_special - len(vocab)
    touched: set[int] = set()

    for merge_i in range(total_merges):
        if verbose_every and (merge_i + 1) % verbose_every == 0:
            print(f"Doing merge {merge_i + 1} out of {total_merges}")

        best_pk, _best_count = heap_pop_best()

        affected = pair_to_words.pop(best_pk, None)
        if not affected:
            # stale heap entry
            pair_counts.pop(best_pk, None)
            maybe_rebuild_heap()
            continue

        a, b = _unpack_pair(best_pk)
        bytes_1 = vocab[a]
        bytes_2 = vocab[b]

        new_token = len(vocab)
        new_bytes = bytes_1 + bytes_2
        vocab.append(new_bytes)
        vocab_key.append(revlex_key(new_bytes))
        merges.append((bytes_1, bytes_2))

        touched.clear()

        for wid in affected:
            wcount = counts[wid]
            if wcount == 0:
                continue  # deactivated word id

            old_word = words[wid]
            if old_word is None:
                continue

            new_word, changed = _transform_word_packed(
                old_word, mv_fmt, a, b, new_token, word_typecode
            )
            if not changed:
                continue

            new_id = word_to_id.get(new_word)
            created = False
            if new_id is None:
                new_id = len(words)
                word_to_id[new_word] = new_id
                words.append(new_word)
                counts.append(0)
                created = True

            counts[new_id] += wcount

            # Deactivate old
            counts[wid] = 0
            words[wid] = None
            word_to_id.pop(old_word, None)

            # Update global pair_counts
            adjust_pair_counts(old_word, -wcount, touched, best_pk)
            adjust_pair_counts(new_word, +wcount, touched, best_pk)

            if created:
                add_word_to_pairs(new_id, new_word, best_pk)

        # best pair is now dead
        pair_counts.pop(best_pk, None)
        entry_finder.pop(best_pk, None)

        for pk in touched:
            c = pair_counts.get(pk, 0)
            if c <= 0:
                pair_counts.pop(pk, None)
                entry_finder.pop(pk, None)
                pair_to_words.pop(pk, None)
            else:
                heap_set(pk, c)

        maybe_rebuild_heap()

    # Add special tokens at the end
    for st in special_tokens:
        vocab.append(st.encode("utf-8"))

    vocab_dict = {i: tok for i, tok in enumerate(vocab)}
    return vocab_dict, merges


def main():
    import json
    import tracemalloc

    Tiny_train_set = "./data/TinyStoriesV2-GPT4-train.txt"
    Tiny_validation_set = "./data/TinyStoriesV2-GPT4-valid.txt"
    tiny_set = "./data/tiny.txt"
    OWT_train_set = "./data/owt_train.txt"
    OWT_validation_set = "./data/owt_valid.txt"
    special_tokens = ["<|endoftext|>"]

    tracemalloc.start()

    vocab, merges = train_bpe(
        OWT_train_set,
        32000,
        special_tokens,
        num_processes=None,  # set to 1/2/4 to reduce RAM; or env BPE_NUM_PROCESSES
        min_count=1,         # try 2 to drop singletons and save lots of RAM
    )

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory usage (tracemalloc): {peak / 1024 / 1024:.2f} MB")

    longest_token = max(vocab.values(), key=len)
    print(f"Longest token ({len(longest_token)} bytes): {longest_token}")

    vocab_serializable = {str(k): v.hex() for k, v in vocab.items()}
    with open("vocab_OWT.json", "w") as f:
        json.dump(vocab_serializable, f, indent=2)

    with open("merges_OWT.txt", "w") as f:
        for b1, b2 in merges:
            f.write(f"{b1.hex()} {b2.hex()}\n")

    print(f"Saved vocab ({len(vocab)} tokens) to vocab_OWT.json")
    print(f"Saved merges ({len(merges)} merges) to merges_OWT.txt")


if __name__ == "__main__":
    profile = True
    if profile:
        import cProfile
        import pstats

        with cProfile.Profile() as pr:
            main()

        stats = pstats.Stats(pr)
        stats.sort_stats("tottime").print_stats()
    else:
        main()
