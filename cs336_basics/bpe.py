### THIS VERSION IS CHATGPT GENERATED, to optimize the code a bit
### 1dadceff963aea065bc858e3368c7316a48c15fc is the last commit written (mostly) by me for this file 
import multiprocessing as mp
import regex as re
from collections import defaultdict, Counter
import heapq

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
# Fast in-Python merge transform
# --------------------------------------------------------------------------------------

def _transform_word_fast(word: tuple[int, ...], a: int, b: int, new_token: int) -> tuple[int, ...]:
    """Replace occurrences of (a,b) left-to-right in a token-id sequence."""
    out: list[int] = []
    append = out.append
    i = 0
    n = len(word)
    while i < n:
        if i + 1 < n and word[i] == a and word[i + 1] == b:
            append(new_token)
            i += 2
        else:
            append(word[i])
            i += 1
    return tuple(out)


# --------------------------------------------------------------------------------------
# Pretokenization (optimized to avoid tuple(...) per match)
# --------------------------------------------------------------------------------------

# GPT-2 regex (as required by the assignment handout)
_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(_PAT)

_SPECIAL_RE_CACHE: dict[tuple[str, ...], re.Pattern] = {}


def _get_special_re(special_tokens: list[str]) -> re.Pattern | None:
    """Compile (and cache) a regex that matches any special token."""
    if not special_tokens:
        return None
    key = tuple(sorted(special_tokens, key=len, reverse=True))
    pat = _SPECIAL_RE_CACHE.get(key)
    if pat is None:
        # Longer-first avoids prefix issues if overlapping specials exist.
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

    Removes special tokens by splitting the stream around them without building
    an intermediate documents list.
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
    Compatibility wrapper: returns dict[tuple[int,...], int] like your original code.
    """
    counts_b = pretokenize_bytes(text, special_tokens)
    return {tuple(k): v for k, v in counts_b.items()}


def pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> dict[bytes, int]:
    """
    Worker function: read the chunk and pretokenize it.

    Returns dict[bytes,int] to keep keys compact across process boundaries.
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize_bytes(chunk, special_tokens)


def parallel_pretokenize_bytes(
    input_path: str,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> dict[bytes, int]:
    """
    Parallel pretokenize returning dict[bytes,int].

    Uses streaming aggregation to reduce peak memory.
    """
    if num_processes is None:
        num_processes = mp.cpu_count()

    # IMPORTANT: don't hard-code <|endoftext|>; use provided specials.
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
    """
    Backwards-compatible wrapper returning dict[tuple[int,...],int].
    """
    counts_b = parallel_pretokenize_bytes(input_path, special_tokens, num_processes=num_processes)
    return {tuple(k): v for k, v in counts_b.items()}


# --------------------------------------------------------------------------------------
# BPE training
# --------------------------------------------------------------------------------------

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train byte-level BPE.

    Returns:
      vocab: dict[token_id -> token_bytes]
      merges: list[(bytes_1, bytes_2)] in merge order
    """
    # --- 1) Pretokenize (bytes keys), then convert to token-id tuples once per unique token ---
    word_counts_b: dict[bytes, int] = parallel_pretokenize_bytes(input_path, special_tokens)

    words: list[tuple[int, ...] | None] = [tuple(wb) for wb in word_counts_b.keys()]
    counts: list[int] = list(word_counts_b.values())
    word_to_id: dict[tuple[int, ...], int] = {w: i for i, w in enumerate(words)}  # active only
    del word_counts_b

    # --- 2) Vocab + tie-break keys (dense ids) ---
    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    vocab_key: list[bytes] = [revlex_key(tok) for tok in vocab]
    merges: list[tuple[bytes, bytes]] = []

    # Safety for packed pairs
    if vocab_size - 1 >= (1 << _SHIFT):
        raise ValueError(f"vocab_size={vocab_size} exceeds packing capacity; increase _SHIFT.")

    # --- 3) pair_counts and pair_to_words ---
    pair_counts: dict[int, int] = defaultdict(int)   # pair_key -> total frequency
    pair_to_words: dict[int, set[int]] = {}          # pair_key -> {word_id} (presence only)

    for wid, (w, c) in enumerate(zip(words, counts)):
        if c == 0 or w is None or len(w) < 2:
            continue
        it = iter(w)
        prev = next(it)
        seen_pairs: set[int] = set()
        for tok in it:
            pk = _pack_pair(prev, tok)
            pair_counts[pk] += c
            if pk not in seen_pairs:
                pair_to_words.setdefault(pk, set()).add(wid)
                seen_pairs.add(pk)
            prev = tok

    # --- 4) Heap with entry_finder + periodic rebuild ---
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

    # --- 5) Single-pass updates per word (counts + pair_to_words) ---
    def _remove_word_from_structures(
        wid: int,
        seq: tuple[int, ...] | None,
        wcount: int,
        touched: set[int],
        skip_pk: int,
    ) -> None:
        """
        Remove an *active* word id from pair_to_words and subtract its contribution
        from pair_counts. `skip_pk` is the current merge pair (already popped from pair_to_words).
        """
        if seq is None or len(seq) < 2:
            return
        it = iter(seq)
        prev = next(it)

        seen: set[int] = set()
        for tok in it:
            pk = _pack_pair(prev, tok)

            # pair_counts counts *occurrences* (not unique), so always update.
            if pk != skip_pk:
                pair_counts[pk] -= wcount
                touched.add(pk)

            # pair_to_words is presence-only, so update once per pk.
            if pk != skip_pk and pk not in seen:
                s = pair_to_words.get(pk)
                if s is not None:
                    s.discard(wid)
                    if not s:
                        pair_to_words.pop(pk, None)
                seen.add(pk)

            prev = tok

    def _add_word_to_structures(
        wid: int,
        seq: tuple[int, ...],
        wcount: int,
        touched: set[int],
        add_to_pair_to_words: bool,
        skip_pk: int,
    ) -> None:
        """
        Add a word's contribution to pair_counts. Optionally add wid to pair_to_words
        (only needed when the word is newly created).
        """
        if len(seq) < 2:
            return
        it = iter(seq)
        prev = next(it)

        if add_to_pair_to_words:
            seen: set[int] = set()
            for tok in it:
                pk = _pack_pair(prev, tok)
                if pk != skip_pk:
                    pair_counts[pk] += wcount
                    touched.add(pk)
                    if pk not in seen:
                        pair_to_words.setdefault(pk, set()).add(wid)
                        seen.add(pk)
                prev = tok
        else:
            for tok in it:
                pk = _pack_pair(prev, tok)
                if pk != skip_pk:
                    pair_counts[pk] += wcount
                    touched.add(pk)
                prev = tok

    # --- 6) Merge loop ---
    target_vocab_no_special = vocab_size - len(special_tokens)
    total_merges = target_vocab_no_special - len(vocab)

    touched: set[int] = set()

    for merge_i in range(total_merges):
        if (merge_i + 1) % 100 == 0:
            print(f"Doing merge {merge_i + 1} out of {total_merges}")

        best_pk, _best_count = heap_pop_best()

        affected = pair_to_words.pop(best_pk, None)
        if not affected:
            # stale heap entry; drop and continue
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
                continue

            old_word = words[wid]
            if old_word is None:
                continue

            new_word = _transform_word_fast(old_word, a, b, new_token)

            # Get/create word id for transformed word
            new_id = word_to_id.get(new_word)
            add_pairs = False
            if new_id is None:
                new_id = len(words)
                word_to_id[new_word] = new_id
                words.append(new_word)
                counts.append(0)
                add_pairs = True

            counts[new_id] += wcount

            # Deactivate old word id
            counts[wid] = 0
            words[wid] = None
            word_to_id.pop(old_word, None)

            # Update global structures
            _remove_word_from_structures(wid, old_word, wcount, touched, best_pk)
            _add_word_to_structures(new_id, new_word, wcount, touched, add_pairs, best_pk)

        # best pair is now dead
        pair_counts.pop(best_pk, None)
        entry_finder.pop(best_pk, None)

        # Update heap priorities once per touched pair (after all words processed)
        for pk in touched:
            c = pair_counts.get(pk, 0)
            if c <= 0:
                pair_counts.pop(pk, None)
                entry_finder.pop(pk, None)
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

    vocab, merges = train_bpe(OWT_validation_set, 10000, special_tokens)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

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
