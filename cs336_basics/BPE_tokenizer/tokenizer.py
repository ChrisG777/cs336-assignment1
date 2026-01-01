### ChatGPT optimized version
### last human commit for this file was 9d6c0af8fd7e48bdbc58374104ae5f038e836066

import json
import heapq
from typing import Iterable, Iterator
import regex as re

# --------------------------------------------------------------------------------------
# Pair packing for dict keys (faster + smaller than tuple[int,int])
# --------------------------------------------------------------------------------------

# Must be > bit_length(max_token_id). 20 supports up to ~1M tokens.
_SHIFT = 20
_MASK = (1 << _SHIFT) - 1


def _pack_pair(a: int, b: int) -> int:
    return (a << _SHIFT) | b


# --------------------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------------------


class Tokenizer:
    # GPT-2 regex (as in the handout)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    # For short “words”, repeated linear scans beat heap overhead.
    _SMALL_WORD_MAX = 64

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # Keep a private copy (don’t mutate caller’s dict)
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)

        # Cache for encode_word (keyed by pretoken string)
        self.cache: dict[str, list[int]] = {}

        # -----------------------------
        # Build token <-> id mappings
        # -----------------------------
        token_to_id: dict[bytes, int] = {tok: tid for tid, tok in self.vocab.items()}

        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []

        # Ensure special tokens exist in vocab (append if missing)
        if self.special_tokens:
            max_id = max(self.vocab) if self.vocab else -1
            for tok in self.special_tokens:
                tb = tok.encode("utf-8")
                if tb not in token_to_id:
                    max_id += 1
                    self.vocab[max_id] = tb
                    token_to_id[tb] = max_id

        self._token_to_id = token_to_id

        # Build dense id->bytes table (supports non-0..N-1 ids too)
        max_id = max(self.vocab) if self.vocab else -1
        id_to_token = [b""] * (max_id + 1)
        for tid, tok in self.vocab.items():
            id_to_token[tid] = tok
        self._id_to_token = id_to_token

        # ---------------------------------------------------------
        # CRITICAL FIX:
        # Precompute byte_value -> token_id from vocab (NOT identity)
        # ---------------------------------------------------------
        byte_to_id = [0] * 256
        for b in range(256):
            tid = token_to_id.get(bytes([b]))
            if tid is None:
                raise ValueError(f"Vocab missing single-byte token for byte {b}.")
            byte_to_id[b] = tid
        self._byte_to_id = byte_to_id

        # ------------------------------------------
        # Precompute merges in *ID space* for speed
        # ------------------------------------------
        pair_rank: dict[int, int] = {}
        new_id_by_rank: list[int] = []

        # rank is merge order (0 is highest priority)
        for rank, (b1, b2) in enumerate(self.merges):
            a = token_to_id[b1]
            b = token_to_id[b2]
            pair_rank[_pack_pair(a, b)] = rank

            merged_bytes = b1 + b2
            new_id = token_to_id.get(merged_bytes)
            if new_id is None:
                # This should not happen if vocab/merges are consistent (GPT-2 or your trainer output).
                raise ValueError(f"Merge creates token {merged_bytes!r} that is not present in vocab.")
            new_id_by_rank.append(new_id)

        self._pair_rank = pair_rank
        self._new_id_by_rank = new_id_by_rank

        # Regex precompile
        self._pat = re.compile(self.PAT)

        # ------------------------------------------
        # Special token handling (fast paths)
        # ------------------------------------------
        if self.special_tokens:
            # Longer-first prevents overlapping-prefix bugs
            self.special_tokens.sort(key=len, reverse=True)

            self._special_to_id: dict[str, int] = {
                tok: token_to_id[tok.encode("utf-8")] for tok in self.special_tokens
            }

            if len(self.special_tokens) == 1:
                # Single special token hot path: use str.find / str.split
                self._single_special = self.special_tokens[0]
                self._single_special_id = self._special_to_id[self._single_special]
                self._special_re = None
                self._max_special_len = len(self._single_special)
            else:
                pat = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
                self._special_re = re.compile(pat)
                self._single_special = None
                self._single_special_id = None
                self._max_special_len = max(len(t) for t in self.special_tokens)
        else:
            self._special_to_id = {}
            self._single_special = None
            self._single_special_id = None
            self._special_re = None
            self._max_special_len = 0

    # -----------------------------
    # IO helpers
    # -----------------------------
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        with open(vocab_filepath, "r") as f:
            vocab_serialized = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in vocab_serialized.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                b1_hex, b2_hex = parts
                merges.append((bytes.fromhex(b1_hex), bytes.fromhex(b2_hex)))

        return cls(vocab, merges, special_tokens or [])

    # -----------------------------
    # Public API
    # -----------------------------
    def decode(self, ids: list[int]) -> str:
        b = b"".join(self._id_to_token[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        out_extend = out.extend
        out_append = out.append

        pat_findall = self._pat.findall
        encode_word = self.encode_word

        if not self.special_tokens:
            for piece in pat_findall(text):
                out_extend(encode_word(piece))
            return out

        # Single special token fast path
        if self._single_special is not None:
            special = self._single_special
            sid = self._single_special_id

            parts = text.split(special)
            last = len(parts) - 1
            for i, part in enumerate(parts):
                if part:
                    for piece in pat_findall(part):
                        out_extend(encode_word(piece))
                if i != last:
                    out_append(sid)
            return out

        # Multiple special tokens
        split = self._special_re.split  # type: ignore[union-attr]
        special_to_id = self._special_to_id
        for part in split(text):
            if not part:
                continue
            sid = special_to_id.get(part)
            if sid is not None:
                out_append(sid)
            else:
                for piece in pat_findall(part):
                    out_extend(encode_word(piece))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Streaming encoder that matches `encode("".join(iterable))` exactly, while keeping memory bounded.

        Why this is non-trivial:
          - The GPT-2 regex can tokenize trailing whitespace differently at end-of-string.
          - Special tokens must be treated as hard boundaries.
          - Chunks may split inside a regex-token or a special token.

        This implementation keeps small buffers to ensure correctness.
        """
        encode_word = self.encode_word
        pat_finditer = self._pat.finditer

        # -----------------------------
        # Helper: stream regex pretokenization with a carry buffer
        # -----------------------------
        pbuf = ""

        def feed_pbuf(s: str, *, final: bool) -> Iterator[int]:
            nonlocal pbuf
            if not s and (not final or not pbuf):
                return
            pbuf = pbuf + s
            if not pbuf:
                return

            last_start = None
            for m in pat_finditer(pbuf):
                end = m.end()
                # If the match reaches the end and we're not at a hard boundary, defer it.
                if (not final) and end == len(pbuf):
                    last_start = m.start()
                    break
                for tid in encode_word(m.group(0)):
                    yield tid

            if last_start is None:
                pbuf = ""
            else:
                pbuf = pbuf[last_start:]

        # -----------------------------
        # No special tokens: just regex streaming
        # -----------------------------
        if not self.special_tokens:
            for chunk in iterable:
                if chunk:
                    yield from feed_pbuf(chunk, final=False)
            if pbuf:
                yield from feed_pbuf("", final=True)
            return

        # -----------------------------
        # With special tokens: stream-scan for specials, flush regex buffer at boundaries
        # -----------------------------
        sbuf = ""
        max_keep = max(0, self._max_special_len - 1)

        # Single special fast path: str.find
        if self._single_special is not None:
            special = self._single_special
            sid = self._single_special_id

            for chunk in iterable:
                if not chunk:
                    continue
                sbuf += chunk

                while True:
                    idx = sbuf.find(special)
                    if idx == -1:
                        break

                    # Text before special is a segment boundary -> flush regex buffer
                    if idx:
                        yield from feed_pbuf(sbuf[:idx], final=True)
                    else:
                        if pbuf:
                            yield from feed_pbuf("", final=True)

                    yield sid
                    sbuf = sbuf[idx + len(special) :]

                # Emit a safe prefix that cannot start a special token
                if max_keep and len(sbuf) > max_keep:
                    safe_end = len(sbuf) - max_keep
                    yield from feed_pbuf(sbuf[:safe_end], final=False)
                    sbuf = sbuf[safe_end:]

            # End-of-stream flush
            if sbuf:
                yield from feed_pbuf(sbuf, final=True)
            elif pbuf:
                yield from feed_pbuf("", final=True)
            return

        # Multiple specials: regex search
        special_re = self._special_re  # type: ignore[assignment]
        special_to_id = self._special_to_id

        for chunk in iterable:
            if not chunk:
                continue
            sbuf += chunk

            while True:
                m = special_re.search(sbuf)
                if m is None:
                    break

                start, end = m.span()

                if start:
                    yield from feed_pbuf(sbuf[:start], final=True)
                else:
                    if pbuf:
                        yield from feed_pbuf("", final=True)

                yield special_to_id[m.group(0)]
                sbuf = sbuf[end:]

            if max_keep and len(sbuf) > max_keep:
                safe_end = len(sbuf) - max_keep
                yield from feed_pbuf(sbuf[:safe_end], final=False)
                sbuf = sbuf[safe_end:]

        # Final flush
        if sbuf:
            yield from feed_pbuf(sbuf, final=True)
        elif pbuf:
            yield from feed_pbuf("", final=True)

    # -----------------------------
    # Core: BPE encode a single pre-token "word"
    # -----------------------------
    def encode_word(self, text: str) -> list[int]:
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        byte_to_id = self._byte_to_id
        pair_rank = self._pair_rank
        new_id_by_rank = self._new_id_by_rank
        pack_pair = _pack_pair
        INF = 1 << 60

        # Start from *byte tokens*, but mapped through vocab -> id (NOT raw 0..255)
        tb = text.encode("utf-8")
        ids = [byte_to_id[b] for b in tb]
        n = len(ids)

        if n < 2:
            self.cache[text] = ids
            return ids

        # -----------------------------
        # Small word: repeated scan
        # -----------------------------
        if n <= self._SMALL_WORD_MAX:
            while True:
                best_rank = INF
                best_pk = -1

                prev = ids[0]
                for j in range(1, len(ids)):
                    curr = ids[j]
                    r = pair_rank.get(pack_pair(prev, curr))
                    if r is not None and r < best_rank:
                        best_rank = r
                        best_pk = pack_pair(prev, curr)
                    prev = curr

                if best_rank == INF:
                    break

                a = best_pk >> _SHIFT
                b = best_pk & _MASK
                merged_id = new_id_by_rank[best_rank]

                out: list[int] = []
                out_append = out.append

                i = 0
                L = len(ids)
                while i < L:
                    if i + 1 < L and ids[i] == a and ids[i + 1] == b:
                        out_append(merged_id)
                        i += 2
                    else:
                        out_append(ids[i])
                        i += 1

                ids = out
                if len(ids) < 2:
                    break

            self.cache[text] = ids
            return ids

        # -----------------------------
        # Long word: heap + linked list
        # -----------------------------
        prev = [i - 1 for i in range(n)]
        nxt = [i + 1 for i in range(n)]
        nxt[-1] = -1

        heap: list[tuple[int, int]] = []
        heappush = heapq.heappush
        heappop = heapq.heappop

        for i in range(n - 1):
            r = pair_rank.get(pack_pair(ids[i], ids[i + 1]))
            if r is not None:
                heappush(heap, (r, i))

        alive = n
        while heap and alive > 1:
            r, i = heappop(heap)
            j = nxt[i]
            if j == -1:
                continue

            pk = pack_pair(ids[i], ids[j])
            if pair_rank.get(pk) != r:
                continue  # stale heap entry

            # merge i and j into i
            ids[i] = new_id_by_rank[r]

            k = nxt[j]
            nxt[i] = k
            if k != -1:
                prev[k] = i
            alive -= 1

            p = prev[i]
            if p != -1:
                rr = pair_rank.get(pack_pair(ids[p], ids[i]))
                if rr is not None:
                    heappush(heap, (rr, p))
            if k != -1:
                rr = pair_rank.get(pack_pair(ids[i], ids[k]))
                if rr is not None:
                    heappush(heap, (rr, i))

        out: list[int] = []
        out_append = out.append
        i = 0
        while i != -1:
            out_append(ids[i])
            i = nxt[i]

        self.cache[text] = out
        return out


def tokenize_dataset(
    dataset_path: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    special_tokens: list[str] | None = None,
    dtype=None,
) -> dict:
    """
    Biggest speed win: DON'T iterate token-by-token if you're building a list anyway.
    Extend with per-line encoded lists.
    """
    import numpy as np

    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    if dtype is None:
        dtype = np.uint16

    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    print(f"Tokenizing {dataset_path}...")

    all_tokens: list[int] = []
    all_tokens_extend = all_tokens.extend

    with open(dataset_path, "r") as f:
        for line in f:
            all_tokens_extend(tokenizer.encode(line))

    print(f"Total tokens: {len(all_tokens)}")

    max_token_id = max(all_tokens) if all_tokens else 0
    print(f"Max token ID: {max_token_id}")

    if dtype == np.uint16 and max_token_id > 65535:
        print(f"WARNING: Max token ID {max_token_id} exceeds uint16 range. Using uint32.")
        dtype = np.uint32
    elif dtype == np.uint8 and max_token_id > 255:
        print(f"WARNING: Max token ID {max_token_id} exceeds uint8 range. Using uint16.")
        dtype = np.uint16

    token_array = np.array(all_tokens, dtype=dtype)
    np.save(output_path, token_array)

    file_size = token_array.nbytes
    print(f"Saved to {output_path}")
    print(f"Array shape: {token_array.shape}, dtype: {token_array.dtype}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    return {
        "total_tokens": len(all_tokens),
        "file_size_bytes": file_size,
        "max_token_id": max_token_id,
        "dtype": str(dtype),
    }


def main():
    """Main function for tokenizing datasets (can be profiled)."""
    import tracemalloc
    
    # Configuration
    dataset_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_path = "vocab.json"
    merges_path = "merges.txt"
    output_path = "./TinyStoriesV2-GPT4-valid_tokens.npy"
    
    print("=" * 60)
    print(f"Tokenizing: {dataset_path}")
    print("=" * 60)
    
    tracemalloc.start()
    
    stats = tokenize_dataset(
        dataset_path=dataset_path,
        vocab_path=vocab_path,
        merges_path=merges_path,
        output_path=output_path,
    )
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"\nPeak memory usage (tracemalloc): {peak_mem / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Tokenize datasets with optional profiling")
    parser.add_argument("--profile", action="store_true", help="Enable cProfile profiling")
    parser.add_argument("--profile-output", type=str, default=None, 
                        help="Save profile stats to file (optional)")
    args = parser.parse_args()
    
    if args.profile:
        import cProfile
        import pstats
        
        print("Running with cProfile enabled...")
        print()
        
        with cProfile.Profile() as pr:
            main()
        
        print("\n" + "=" * 60)
        print("PROFILING RESULTS (sorted by total time)")
        print("=" * 60)
        
        stats = pstats.Stats(pr)
        stats.sort_stats("tottime").print_stats(50)  # Top 50 functions
        
        if args.profile_output:
            with open(args.profile_output, "w") as f:
                stats_file = pstats.Stats(pr, stream=f)
                stats_file.sort_stats("tottime").print_stats()
            print(f"\nFull profile saved to: {args.profile_output}")
    else:
        import time
        start_time = time.time()
        main()
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time:.2f} seconds")
