import multiprocessing as mp
import regex as re
from collections import defaultdict, Counter
from .pretokenization_example import find_chunk_boundaries
from pprint import pprint
from heapq import heappush, heappop
from functools import total_ordering

@total_ordering
class ReversedBytes:
    """Wrapper for bytes that reverses comparison order (larger bytes compare as smaller)"""
    __slots__ = ['b']
    def __init__(self, b: bytes):
        self.b = b
    def __lt__(self, other):
        return self.b > other.b  # Reversed: larger bytes come first in min-heap
    def __eq__(self, other):
        return self.b == other.b

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    word_counts : dict[tuple[int, ...], int]= parallel_pretokenize(input_path, special_tokens)
    vocab : dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}
    vocab_negated : dict[int, ReversedBytes] = {idx: ReversedBytes(bytes([idx])) for idx in range(256)}
    merges : list[tuple[bytes, bytes]]  = []
    pair_counts : Counter[tuple[int, int]] = Counter() # tracks how many times each pair of tokens shows up 
    pair_counts_heap = [] # (negative count, negative underlying bytes, token pair) to get the token pair with largest count and highest lexicographical order. The pair is just there so that we actually know what pair it corresponds to when we pop it
    # we won't bother to get rid of stale entries from this heap. Instead, when we pop, we'll just cross check with pair_counts to see if the count is stale
    # i.e., the invariant we'll keep is that all up to date pair counts are in the heap, so if an unstale count is the max, then it really must be higher than all the up to date counts
    pair_to_words : defaultdict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set) # tracks in which words each pair of tokens shows up 

    # initialize everything 
    for word, count in word_counts.items():
        pair_counts.update(scale_counter(pair_counts_in_word(word), count))
        for token, next_token in zip(word[:-1], word[1:]):
            pair_to_words[(token, next_token)].add(word)

    def add_to_heap(pair, count):
        # Use tuple comparison for lexicographic tiebreaker
        # ReversedBytes reverses comparison so min-heap gives max lexicographic order
        reversed_tuple = (vocab_negated[pair[0]], vocab_negated[pair[1]])
        heappush(pair_counts_heap, (-count, reversed_tuple, pair))
    
    for pair, count in pair_counts.items(): 
        add_to_heap(pair, count) 

    # during the merging process, on each merge we 
    # 1. look at the highest count pair, broken by largest lexicographical order
    # 2. add that to merges, and add the new token to vocab
    # 3. go through all words with that pair. 
    # a. Get the old word, and get the new word
    # b. Update word_counts with the new words, delete the old words
    # c. Update pair_counts by subtracting the pair counts for the old word, adding the pair counts for the new word (can use .update and .subtract on Counters). Update the heap by pushing on the new pair with the new count
    # d. Update pair_to_words by removing the word from all the old pairs in it, and adding the word for all the new pairs in it
    while len(vocab) < vocab_size - len(special_tokens):
        # 1. look at the highest count pair, broken by largest lexicographical order
        best_pair = None
        while pair_counts_heap:
            neg_count, _, pair = heappop(pair_counts_heap)
            count = -neg_count
            if pair in pair_counts and pair_counts[pair] == count:
                # up to date pair and count 
                best_pair = pair
                break
        if best_pair is None:
            raise ValueError("should be getting some pair from the heap")
        # best_pair = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]] + vocab[p[1]]))

        # 2. add that to merges, and add the new token to vocab
        bytes_1 = vocab[best_pair[0]]
        bytes_2 = vocab[best_pair[1]]
        new_token = len(vocab)
        vocab[new_token] = bytes_1 + bytes_2
        vocab_negated[new_token] = ReversedBytes(bytes_1 + bytes_2)
        merges.append((bytes_1, bytes_2))

        # 3. go through all words with that pair. 
        for word in list(pair_to_words[best_pair]): # list to get a copy
            num_word_occurrences = word_counts[word] 

            # a. Get the old word, and get the new word
            old_word = word
            new_word = transform_word(word, best_pair, new_token)
            # b. Update word_counts with the new words, delete the old words
            word_counts[new_word] = word_counts.get(new_word, 0) + word_counts[old_word] # have to add in case two old words map to the same new word
            del word_counts[old_word]
            

            # c. Update pair_counts by subtracting the pair counts for the old word, adding the pair counts for the new word (can use .update and .subtract on Counters)
            
            pair_counts_old = scale_counter(pair_counts_in_word(old_word), num_word_occurrences)
            pair_counts_new = scale_counter(pair_counts_in_word(new_word), num_word_occurrences)

            pair_counts.update(pair_counts_new)
            pair_counts.subtract(pair_counts_old)
            for pair in pair_counts_new.keys() | pair_counts_old.keys(): 
                if pair_counts[pair] > 0: 
                    add_to_heap(pair, pair_counts[pair])

            # d. Update pair_to_words by removing the word from all the old pairs in it, and adding the word for all the new pairs in it
            for token, next_token in zip(old_word[:-1], old_word[1:]):
                pair_to_words[(token, next_token)].discard(old_word) # discard instead of remove since it's idempotent
            for token, next_token in zip(new_word[:-1], new_word[1:]):
                pair_to_words[(token, next_token)].add(new_word)

    # add in the special tokens
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode('utf-8')
    return vocab, merges

def scale_counter(counter: Counter, multiplier: int) -> Counter:
    return Counter({key: count * multiplier for key, count in counter.items()})

def pair_counts_in_word(word: tuple[int, ...]) -> Counter[tuple[int, int]]:
    """
    Given a word, outputs a counter of how many times each pair of tokens apppears in that word
    """
    count : Counter[tuple[int, int]] = Counter() 
    for token, next_token in zip(word[:-1], word[1:]):
        count[(token, next_token)] += 1
    return count


def transform_word(word: tuple[int, ...], pair: tuple[int, int], new_token) -> tuple[int, ...]:
    """
    transforms the word, replacing the old byte pair by the new token left to right
    """
    new_word = []
    i = 0
    while i < len(word):
        if i < len(word)-1 and (word[i], word[i+1]) == pair:
            new_word.append(new_token)
            i += 2
        else:
            new_word.append(word[i]) 
            i += 1
    return tuple(new_word)



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
    num_processes = None
    Tiny_train_set = "./data/TinyStoriesV2-GPT4-train.txt"
    Tiny_validation_set = "./data/TinyStoriesV2-GPT4-valid.txt"
    tiny_set = "./data/tiny.txt"
    special_tokens = ["<|endoftext|>"]

    # word_counts = parallel_pretokenize(Tiny_validation_set, special_tokens, num_processes)
    # print(word_counts)

    vocab, merges = train_bpe(Tiny_train_set, 10000, special_tokens)
    pprint(merges[:10])

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
    