import json
from typing import Iterable, Iterator
import regex as re


class Tokenizer:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        # --- vocab / ids ---
        self.vocab = vocab

        # (Optional) ensure special tokens are in vocab
        if special_tokens:
            # Avoid O(|vocab|) "x in vocab.values()" for every special token
            values_set = set(self.vocab.values())
            for tok in special_tokens:
                b = tok.encode("utf-8")
                if b not in values_set:
                    self.vocab[len(self.vocab)] = b
                    values_set.add(b)

        self.vocab_index = {b: idx for idx, b in self.vocab.items()}

        # --- merges ---
        self.merges = merges
        self.merges_index = {(b1, b2): i for i, (b1, b2) in enumerate(merges)}

        # --- cache ---
        self.cache: dict[str, list[int]] = {}

        # --- regex precompile (big speed win) ---
        self._pat = re.compile(self.PAT)

        # --- special token handling ---
        self.special_tokens = special_tokens or []
        if self.special_tokens:
            # Longer-first avoids prefix issues if overlapping specials exist.
            key = tuple(sorted(self.special_tokens, key=len, reverse=True))
            self.special_tokens = list(key)

            self._special_to_id: dict[str, int] = {
                tok: self.vocab_index[tok.encode("utf-8")] for tok in self.special_tokens
            }

            if len(self.special_tokens) == 1:
                # Hot-path: single special token (e.g. "<|endoftext|>") -> use str.split
                self._single_special = self.special_tokens[0]
                self._single_special_id = self._special_to_id[self._single_special]
                self._special_re = None
            else:
                self._single_special = None
                self._single_special_id = None
                pat = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
                self._special_re = re.compile(pat)
        else:
            self._special_to_id = {}
            self._single_special = None
            self._single_special_id = None
            self._special_re = None

        # Optional micro-opt for cache-miss path: reuse 1-byte objects
        self._byte_tokens = [bytes([i]) for i in range(256)]

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

    def encode(self, text: str) -> list[int]:
        """
        Faster encode:
          - uses precompiled regex
          - uses findall() (avoids millions of match.group() calls)
          - uses a dict lookup for special tokens
          - fast-path for single special token via str.split
        """
        out: list[int] = []
        out_append = out.append
        out_extend = out.extend

        pat_findall = self._pat.findall
        encode_word = self.encode_word

        if not self.special_tokens:
            for piece in pat_findall(text):
                out_extend(encode_word(piece))
            return out

        # Single special token fast path
        if self._single_special is not None:
            special = self._single_special
            special_id = self._single_special_id

            parts = text.split(special)
            last = len(parts) - 1
            for i, part in enumerate(parts):
                if part:
                    for piece in pat_findall(part):
                        out_extend(encode_word(piece))
                if i != last:
                    out_append(special_id)
            return out

        # Multiple special tokens path
        special_split = self._special_re.split  # type: ignore[union-attr]
        special_to_id = self._special_to_id

        for part in special_split(text):
            if not part:
                continue
            tok_id = special_to_id.get(part)
            if tok_id is not None:
                out_append(tok_id)
            else:
                for piece in pat_findall(part):
                    out_extend(encode_word(piece))
        return out

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Much simpler + faster than your current version.

        Under your stated assumption (“special tokens don’t span lines”), we do NOT need
        the complicated buffer/match-end logic. That logic was costing you a lot.

        Still yields ints one-by-one (same API).
        """
        pat_findall = self._pat.findall
        encode_word = self.encode_word

        # No special tokens
        if not self.special_tokens:
            for chunk in iterable:
                for piece in pat_findall(chunk):
                    yield from encode_word(piece)
            return

        # Single special token fast path
        if self._single_special is not None:
            special = self._single_special
            special_id = self._single_special_id

            for chunk in iterable:
                if special not in chunk:
                    for piece in pat_findall(chunk):
                        yield from encode_word(piece)
                    continue

                parts = chunk.split(special)
                last = len(parts) - 1
                for i, part in enumerate(parts):
                    if part:
                        for piece in pat_findall(part):
                            yield from encode_word(piece)
                    if i != last:
                        yield special_id
            return

        # Multiple special tokens
        special_split = self._special_re.split  # type: ignore[union-attr]
        special_to_id = self._special_to_id
        for chunk in iterable:
            for part in special_split(chunk):
                if not part:
                    continue
                tok_id = special_to_id.get(part)
                if tok_id is not None:
                    yield tok_id
                else:
                    for piece in pat_findall(part):
                        yield from encode_word(piece)

    def decode(self, ids: list[int]) -> str:
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def encode_word(self, text: str) -> list[int]:
        """
        Your BPE logic, with two small speed tweaks:
          - cache.get() fast path
          - reuse singleton 1-byte tokens (avoids tons of tiny bytes allocations on cache misses)
        """
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        merges_index = self.merges_index
        vocab_index = self.vocab_index
        byte_tokens = self._byte_tokens

        text_bytes = text.encode("utf-8")
        current: list[bytes] = [byte_tokens[b] for b in text_bytes]

        # Repeatedly merge the lowest-rank pair
        while True:
            best_rank = 10**18
            best_pair = None

            # Scan all adjacent pairs
            # (Keeping your semantics exactly; if you later want, we can implement a heap-based
            #  O(n log n) BPE which helps when you have lots of cache misses.)
            for i in range(len(current) - 1):
                pair = (current[i], current[i + 1])
                rank = merges_index.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            current = _replace_merged_bytes(current, best_pair)

        final_tokens = [vocab_index[b] for b in current]
        self.cache[text] = final_tokens
        return final_tokens


def _replace_merged_bytes(seq: list[bytes], pair: tuple[bytes, bytes]) -> list[bytes]:
    merged = pair[0] + pair[1]
    out: list[bytes] = []
    i = 0
    n = len(seq)
    a, b = pair
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(seq[i])
            i += 1
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
