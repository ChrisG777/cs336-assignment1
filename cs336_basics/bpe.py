import multiprocessing as mp
import regex as re
from collections import defaultdict, Counter
from .pretokenization_example import find_chunk_boundaries
from pprint import pprint
import heapq

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
    out[1:2*n:2] = inv   # 0x00, inv[0], 0x00, inv[1], ...
    out[2*n] = 0xFF      # terminator so shorter originals sort after longer
    return bytes(out)

# Pack a pair (a,b) into a single int to reduce dict/set overhead.
# SHIFT must be > max token id bits. 20 supports up to ~1M tokens.
_SHIFT = 20
_MASK = (1 << _SHIFT) - 1

def _pack_pair(a: int, b: int) -> int:
    return (a << _SHIFT) | b

def _unpack_pair(p: int) -> tuple[int, int]:
    return p >> _SHIFT, p & _MASK

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # --- 1) Pretokenize ---
    word_counts_dict: dict[tuple[int, ...], int] = parallel_pretokenize(input_path, special_tokens)

    # Convert to ID-based storage so pair_to_words stores ints (fast hash),
    # not whole tuples (slow hash + more memory).
    words: list[tuple[int, ...] | None] = list(word_counts_dict.keys())
    counts: list[int] = list(word_counts_dict.values())
    word_to_id: dict[tuple[int, ...], int] = {w: i for i, w in enumerate(words)}
    del word_counts_dict  # allow GC of big dict

    # --- 2) Vocab as lists (faster than dicts for 0..N-1 ids) ---
    vocab: list[bytes] = [bytes([i]) for i in range(256)]
    vocab_key: list[bytes] = [revlex_key(tok) for tok in vocab]
    merges: list[tuple[bytes, bytes]] = []

    # --- 3) pair_counts and pair_to_words ---
    pair_counts: dict[int, int] = defaultdict(int)   # pair_key -> count
    pair_to_words: dict[int, set[int]] = {}          # pair_key -> {word_id}

    # Build initial counts/mapping without tuple slicing
    for wid, (w, c) in enumerate(zip(words, counts)):
        if c == 0 or w is None or len(w) < 2:
            continue
        it = iter(w)
        prev = next(it)
        seen_pairs = set()
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

    # init heap
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
        # Tune the constants if needed.
        if len(heap) > 4 * len(entry_finder) + 200_000:
            heap[:] = list(entry_finder.values())
            heapify(heap)

    # --- 5) Helpers to update counts and pair_to_words without slicing ---
    def adjust_pair_counts(seq: tuple[int, ...] | None, delta: int, touched: set[int]) -> None:
        if seq is None or len(seq) < 2:
            return
        it = iter(seq)
        prev = next(it)
        for tok in it:
            pk = _pack_pair(prev, tok)
            pair_counts[pk] += delta
            touched.add(pk)
            prev = tok

    def remove_word_from_pairs(wid: int, seq: tuple[int, ...] | None) -> None:
        if seq is None or len(seq) < 2:
            return
        it = iter(seq)
        prev = next(it)
        seen = set()
        for tok in it:
            pk = _pack_pair(prev, tok)
            if pk not in seen:
                s = pair_to_words.get(pk)
                if s is not None:
                    s.discard(wid)
                    if not s:
                        pair_to_words.pop(pk, None)
                seen.add(pk)
            prev = tok

    def add_word_to_pairs(wid: int, seq: tuple[int, ...]) -> None:
        if len(seq) < 2:
            return
        it = iter(seq)
        prev = next(it)
        seen = set()
        for tok in it:
            pk = _pack_pair(prev, tok)
            if pk not in seen:
                pair_to_words.setdefault(pk, set()).add(wid)
                seen.add(pk)
            prev = tok

    # --- 6) Merge loop ---
    target_vocab_no_special = vocab_size - len(special_tokens)
    total_merges = target_vocab_no_special - len(vocab)
    
    for merge_i in range(total_merges):
        if (merge_i + 1) % 100 == 0:
            print(f"Doing merge {merge_i + 1} out of {total_merges}")

        best_pk, best_count = heap_pop_best()

        # If this pair has no words anymore (stale), drop and continue.
        affected = pair_to_words.pop(best_pk, None)
        if not affected:
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

        # Process each affected word id
        for wid in affected:
            wcount = counts[wid]
            if wcount == 0:
                continue

            old_word = words[wid]
            if old_word is None:
                continue

            new_word = _transform_word_fast(old_word, a, b, new_token)

            # Merge counts into existing/new word id
            new_id = word_to_id.get(new_word)
            created_new = False
            if new_id is None:
                new_id = len(words)
                word_to_id[new_word] = new_id
                words.append(new_word)
                counts.append(wcount)
                created_new = True
            else:
                # If somehow reactivating an old word id, re-add to pair sets
                if counts[new_id] == 0:
                    created_new = True
                counts[new_id] += wcount

            # Deactivate old
            counts[wid] = 0
            # Remove old sequence from mapping + free list slot (saves memory)
            word_to_id.pop(old_word, None)
            words[wid] = None

            # Update pair_to_words: remove old id from all its pairs
            remove_word_from_pairs(wid, old_word)

            # If this new_id is newly created/re-activated, add it to pair_to_words
            if created_new:
                add_word_to_pairs(new_id, new_word)

            # Update global pair_counts + heap for touched pairs
            touched: set[int] = set()
            adjust_pair_counts(old_word, -wcount, touched)
            adjust_pair_counts(new_word, +wcount, touched)

            for pk in touched:
                if pk == best_pk:
                    # We're killing best_pk; never need it in heap again.
                    continue
                c = pair_counts.get(pk, 0)
                if c <= 0:
                    pair_counts.pop(pk, None)
                    entry_finder.pop(pk, None)
                else:
                    heap_set(pk, c)

        # best pair should now be gone
        pair_counts.pop(best_pk, None)
        entry_finder.pop(best_pk, None)

        maybe_rebuild_heap()

    # Add special tokens at the end
    for st in special_tokens:
        vocab.append(st.encode("utf-8"))

    # Return as dict[int, bytes] to match your existing interface
    vocab_dict = {i: tok for i, tok in enumerate(vocab)}
    return vocab_dict, merges


# def scale_counter(counter: Counter, multiplier: int) -> Counter:
#     return Counter({key: count * multiplier for key, count in counter.items()})

# def pair_counts_in_word(word: tuple[int, ...]) -> Counter[tuple[int, int]]:
#     """
#     Given a word, outputs a counter of how many times each pair of tokens apppears in that word
#     """
#     count : Counter[tuple[int, int]] = Counter() 
#     for token, next_token in zip(word[:-1], word[1:]):
#         count[(token, next_token)] += 1
#     return count


# def transform_word(word: tuple[int, ...], pair: tuple[int, int], new_token) -> tuple[int, ...]:
#     """
#     transforms the word, replacing the old byte pair by the new token left to right
#     """
#     new_word = []
#     i = 0
#     while i < len(word):
#         if i < len(word)-1 and (word[i], word[i+1]) == pair:
#             new_word.append(new_token)
#             i += 2
#         else:
#             new_word.append(word[i]) 
#             i += 1
#     return tuple(new_word)
def _transform_word_fast(word: tuple[int, ...], a: int, b: int, new_token: int) -> tuple[int, ...]:
    # Replaces occurrences of (a,b) left-to-right.
    out = []
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


def pretokenize_chunk(args):
    """
    Worker function

    reads in the chunk and then calls pretokenize on it
    """
    input_path, start, end, special_tokens = args
    with open(input_path, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize(chunk, special_tokens)

def parallel_pretokenize(input_path: str, special_tokens: list[str], num_processes: int = None) -> dict[tuple[int, ...], int]:
    """
    Reads in text and then basically wraps a bunch of parallel calls to pretokenize
    Aggregates the counts at the end 
    """
    if num_processes is None:
        num_processes = mp.cpu_count()
    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    work_items = [
        (input_path, start, end, special_tokens) 
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with mp.Pool(num_processes) as pool:
        results = pool.map(pretokenize_chunk, work_items)
    
    total_counts = Counter()
    for counts in results:
        for word, count in counts.items():
            total_counts[word] += count
    return dict(total_counts) 
    
def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    """
    pretokenizes the text, generating counts of all of the "words" that appear in the text, counting separate documents separately 
    """
    documents = split_documents(text, special_tokens) # this is taking almost no time 
    word_counts = pretokenize_documents(documents) # main bottleneck
    return word_counts 

def split_documents(text: str, special_tokens: list[str]) -> list[str]:
    """
    Splits up text into a list of the documents delimited by the special tokens
    """
    special_tokens_escaped = [re.escape(token) for token in special_tokens] 
    special_pattern = "|".join(special_tokens_escaped)
    documents = re.split(special_pattern, text)
    return [document for document in documents if document != '']

def pretokenize_documents(documents: list[str]) -> dict[tuple[int, ...], int]:
    """
    helper function that gets called by the full pretokenize function 
    """
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_counts = defaultdict(int)
    for document in documents:
        for match in re.finditer(PAT, document):
            word = match.group()
            word_as_bytes = tuple(word.encode('utf-8'))
            word_counts[word_as_bytes] += 1
            # word_counts[word] += 1 # just useful for debugging / seeing the actual words themselves
    return word_counts

def main():
    import json
    import tracemalloc

    num_processes = None
    Tiny_train_set = "./data/TinyStoriesV2-GPT4-train.txt"
    Tiny_validation_set = "./data/TinyStoriesV2-GPT4-valid.txt"
    tiny_set = "./data/tiny.txt"
    OWT_train_set = "./data/owt_train.txt"
    OWT_validation_set = "./data/owt_valid.txt"
    special_tokens = ["<|endoftext|>"]

    # Track memory usage
    tracemalloc.start()

    # vocab, merges = train_bpe(Tiny_validation_set, 10000, special_tokens)
    vocab, merges = train_bpe(OWT_validation_set, 10000, special_tokens)

    # Get peak memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

    # Find and print the longest token
    longest_token = max(vocab.values(), key=len)
    print(f"Longest token ({len(longest_token)} bytes): {longest_token}")

    # Serialize vocab and merges to disk
    # Vocab: convert bytes to hex strings for JSON compatibility
    vocab_serializable = {str(k): v.hex() for k, v in vocab.items()}
    with open("vocab_OWT.json", "w") as f:
        json.dump(vocab_serializable, f, indent=2)

    # Merges: save as text file (one merge per line, hex-encoded)
    with open("merges_OWT.txt", "w") as f:
        for b1, b2 in merges:
            f.write(f"{b1.hex()} {b2.hex()}\n")

    print(f"Saved vocab ({len(vocab)} tokens) to vocab_OWT.json")
    print(f"Saved merges ({len(merges)} merges) to merges_OWT.txt")

if __name__ == '__main__':
    profile = True 
    if profile:
        import cProfile
        import pstats

        # Run the profiler programmatically
        # This keeps 'bpe.py' as the true __main__, fixing the pickle error
        with cProfile.Profile() as pr:
            main()
        
        # Print the results sorted by total time
        stats = pstats.Stats(pr)
        stats.sort_stats('tottime').print_stats()
        
        # Optional: Save to file
        # stats.dump_stats("profile_results.txt")
    else:
        main()
    