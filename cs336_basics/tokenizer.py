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


if __name__ == '__main__':   
    tokenizer = Tokenizer.from_files("vocab.json", "merges.txt", special_tokens=["<|endoftext|>"])

    # had cursor write some tests for me bc I'm lazy lol

    # Test 3: Encode a file
    print("\n--- Test 3: Encode tiny.txt ---")
    tiny_set = "./data/tiny.txt"
    with open(tiny_set, "r") as f:
        body = f.read()
    tokens = tokenizer.encode(body)
    print(f"File length: {len(body)} chars")
    print(f"Token count: {len(tokens)}")
    print(f"First 20 tokens: {tokens[:20]}")
    
    # Test 4: encode_iterable (streaming)
    print("\n--- Test 4: encode_iterable (streaming) ---")
    with open(tiny_set, "r") as f:
        iterable_tokens = list(tokenizer.encode_iterable(f))
    print(f"Iterable token count: {len(iterable_tokens)}")
    print(f"First 20 tokens: {iterable_tokens[:20]}")
    
    # Test 5: Compare encode vs encode_iterable
    print("\n--- Test 5: encode vs encode_iterable consistency ---")
    if tokens == iterable_tokens:
        print("✓ PASS: encode() and encode_iterable() produce identical results")
    else:
        print("✗ FAIL: Results differ!")
        print(f"  encode() length: {len(tokens)}")
        print(f"  encode_iterable() length: {len(iterable_tokens)}")
        # Find first difference
        for i, (a, b) in enumerate(zip(tokens, iterable_tokens)):
            if a != b:
                print(f"  First difference at index {i}: {a} vs {b}")
                break
    
    # Test 6: Round-trip (if decode is implemented)
    print("\n--- Test 6: Round-trip test ---")
    try:
        decoded = tokenizer.decode(tokens)
        if decoded == body:
            print("✓ PASS: Round-trip successful")
        else:
            print("✗ FAIL: Decoded text doesn't match original")
    except (NotImplementedError, TypeError, AttributeError) as e:
        print(f"(decode not implemented: {type(e).__name__})")
    
    print("\n" + "=" * 60)
    
