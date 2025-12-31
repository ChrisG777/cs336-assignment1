import json
from typing import Iterable, Iterator 
import regex as re

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens : list[str] | None =None):
        self.vocab = vocab 
        if special_tokens:
            for special_token in special_tokens:
                if special_token.encode('utf-8') not in self.vocab.values():
                    self.vocab[len(self.vocab)] = special_token.encode('utf-8')
        self.vocab_index = { b:idx for idx, b in self.vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        
        self.merges_index = {(bytes_1, bytes_2) : i  for i, (bytes_1, bytes_2) in enumerate(merges)} # reverse lookup of the merge order of a pattern of bytes
        self.cache: dict[str, list[int]]= {} # cache the tokenization of words 
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if special_tokens:
            # Longer-first avoids prefix issues if overlapping specials exist.
            key = tuple(sorted(special_tokens, key=len, reverse=True))
            self.special_pattern = "(" + "|".join(re.escape(tok) for tok in key) + ")"
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        """
        method that constructs and return a Tokenizer from a serialized vocabulary and list of merges
        (in the same format that the BPE training code output) and (optionally) a list of special tokens.
        """
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
        Encode an input text into a sequence of token IDs.

        Use re.split to split up special tokens
        Use the same re.finditer to look for words
        Call encode_word on every word
        Return a flattened list of the tokens for each word 
        """
        if self.special_tokens:
            documents = re.split(self.special_pattern, text)
        else:
            documents = [text]
        tokens = []
        for document in documents:
            if self.special_tokens and document in self.special_tokens:
                tokens += [self.vocab_index[document.encode('utf-8')]]
                continue 
            for word in re.finditer(self.PAT, document):
                tokens += self.encode_word(word.group())
        return tokens 

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-eﬀicient tokenization of large files that we cannot directly load into memory.

        Keep a buffer (making sure to process what's left at the end). I'm going to assume that special tokens don't span across multiple lines 

        Call encode on each line, yield_from it 
        """
        buffer = ""
        if self.special_tokens:
            margin = max(len(special_token) for special_token in self.special_tokens)
        else:
            margin = 0
 
        for s in iterable:
            buffer += s

            if self.special_tokens:
                parts = re.split(self.special_pattern, buffer)
            else: 
                parts = [buffer]
            
            buffer = "" # reset the buffer
            
            for i, part in enumerate(parts): 
                if not part:
                    continue

                # check if it's a special token
                if self.special_tokens and part in self.special_tokens:
                    yield self.vocab_index[part.encode('utf-8')]
                    continue

                is_last_part = (i == len(parts) - 1)

                # encode all the matches to words within the part, besides the very last possibly unfinished one 
                matches = list(re.finditer(self.PAT, part)) # can't use findall because we need the match objects
                for j, match in enumerate(matches):
                    if j == len(matches) - 1 and is_last_part and match.end() == len(part):
                        # this match goes to the end of the current buffer. Include it in the next buffer, since it might be unfinished
                        buffer = match.group() + buffer
                        break 
                    yield from self.encode_word(match.group())
        
        # flush whatever's left in the buffer 
        if buffer:
            if self.special_tokens:
                parts = re.split(self.special_pattern, buffer)
            else:
                parts = [buffer]
            for part in parts:
                if self.special_tokens and part in self.special_tokens:
                    yield self.vocab_index[part.encode('utf-8')]
                elif part:
                    for match in re.finditer(self.PAT, part):
                        yield from self.encode_word(match.group())

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text
        """
        bytes = b"".join(self.vocab[id] for id in ids)
        decoded_string = bytes.decode('utf-8', errors="replace")
        return decoded_string 


    def encode_word(self, text: str) -> list[int]: 
        """
        Encodes a single word, with no special tokens and no word breaks within it

        Repeatedly scan for the pair with lowest rank and merge it left to right, staying in bytes

        Then convert to ints 
        """
        if text in self.cache:
            return self.cache[text]
        text_bytes = text.encode('utf-8')
        current_bytes_sequence : list[bytes] = [text_bytes[i:i+1] for i in range(len(text_bytes))]
        while True:
            mergeable = False
            lowest_merging_idx = len(self.merges)
            earliest_merge_pair = None
            for i in range(len(current_bytes_sequence) - 1):
                byte_pair = (current_bytes_sequence[i], current_bytes_sequence[i+1])
                if byte_pair in self.merges_index:
                    mergeable = True
                    cur_merging_idx = self.merges_index[byte_pair]
                    if cur_merging_idx < lowest_merging_idx:
                        lowest_merging_idx = cur_merging_idx
                        earliest_merge_pair = byte_pair
            if not mergeable:
                break 

            current_bytes_sequence = replace_merged_bytes(current_bytes_sequence, earliest_merge_pair) 
            

        # convert the current bytes sequence into the ints
        final_tokens = [self.vocab_index[b] for b in current_bytes_sequence]
        self.cache[text] = final_tokens
        return final_tokens

def replace_merged_bytes(byte_sequence: list[bytes], byte_pair: tuple[bytes, bytes]):
    """
    Returns a new byte sequence with the occurrences of byte_pair replaced by their concatenation, going left to right
    """
    new_bytes_sequence = []
    merged_token_bytes = byte_pair[0] + byte_pair[1]
    i = 0
    while i < len(byte_sequence):
        if i < len(byte_sequence) - 1 and byte_sequence[i] == byte_pair[0] and byte_sequence[i+1] == byte_pair[1]:
            new_bytes_sequence.append(merged_token_bytes)
            i += 2
        else: 
            new_bytes_sequence.append(byte_sequence[i])
            i += 1
    return new_bytes_sequence


def tokenize_n_documents(
    tokenizer: Tokenizer,
    dataset_path: str,
    n: int,
    special_token: str = "<|endoftext|>",
    verbose: bool = True,
) -> tuple[list[int], dict]:
    """
    Written by cur
    Sample and tokenize n documents from a dataset.
    
    Args:
        tokenizer: A Tokenizer instance
        dataset_path: Path to the dataset file (documents separated by special_token)
        n: Number of documents to tokenize
        special_token: The token that separates documents
        verbose: Whether to print progress
        
    Returns:
        tuple of (token_ids, stats) where stats is a dict with:
            - num_documents: number of documents tokenized
            - total_chars: total characters in documents
            - total_bytes: total bytes in documents (UTF-8)
            - total_tokens: total tokens produced
            - bytes_per_token: compression ratio
    """
    with open(dataset_path, "r") as f:
        content = f.read()
    
    # Split by special token to get individual documents
    documents = content.split(special_token)[:n]
    
    all_tokens = []
    total_chars = 0
    total_bytes = 0
    
    for i, doc in enumerate(documents):
        doc = doc.strip()
        if not doc:
            continue
        
        # Add back the special token at the end
        doc_with_special = doc + special_token
        tokens = tokenizer.encode(doc_with_special)
        all_tokens.extend(tokens)
        
        doc_bytes = len(doc_with_special.encode('utf-8'))
        total_chars += len(doc_with_special)
        total_bytes += doc_bytes
        
        if verbose:
            print(f"Document {i+1}: {len(doc)} chars -> {len(tokens)} tokens")
    
    stats = {
        "num_documents": len([d for d in documents if d.strip()]),
        "total_chars": total_chars,
        "total_bytes": total_bytes,
        "total_tokens": len(all_tokens),
        "bytes_per_token": total_bytes / len(all_tokens) if all_tokens else 0,
    }
    
    if verbose:
        print(f"\nTotal tokens: {stats['total_tokens']}")
        print(f"Compression ratio: {stats['bytes_per_token']:.2f} bytes/token")
    
    return all_tokens, stats


def tokenize_dataset(
    dataset_path: str,
    vocab_path: str,
    merges_path: str,
    output_path: str,
    special_tokens: list[str] | None = None,
    dtype = None,  # numpy dtype, default uint16
) -> dict:
    """
    Tokenize an entire dataset and save as a numpy array.
    
    Args:
        dataset_path: Path to the dataset file to tokenize
        vocab_path: Path to the vocabulary JSON file
        merges_path: Path to the merges file
        output_path: Path to save the numpy array (.npy)
        special_tokens: List of special tokens (default: ["<|endoftext|>"])
        dtype: Numpy dtype for the output array (default: uint16)
        
    Returns:
        dict with stats:
            - total_tokens: number of tokens
            - file_size_bytes: size of the saved file
            - max_token_id: maximum token ID (to verify dtype is sufficient)
    """
    import numpy as np
    
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    if dtype is None:
        dtype = np.uint16
    
    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    
    print(f"Tokenizing {dataset_path}...")
    
    # Use encode_iterable for memory efficiency on large files
    all_tokens = []
    with open(dataset_path, "r") as f:
        for token_id in tokenizer.encode_iterable(f):
            all_tokens.append(token_id)
    
    print(f"Total tokens: {len(all_tokens)}")
    
    max_token_id = max(all_tokens) if all_tokens else 0
    print(f"Max token ID: {max_token_id}")
    
    # Check if dtype can hold the max token ID
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
