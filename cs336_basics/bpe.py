import multiprocessing as mp 
import regex as re
from collections import defaultdict, Counter
from pretokenization_example import find_chunk_boundaries

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pass

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

def parallel_pretokenize(input_path: str, special_tokens: list[str], num_processes: int = None) -> dict[tuple[bytes], int]:
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
    
    

def pretokenize(text: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
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

def pretokenize_documents(documents: list[str]) -> dict[tuple[bytes], int]:
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
    Tiny_train_set = "../data/TinyStoriesV2-GPT4-train.txt"
    Tiny_validation_set = "../data/TinyStoriesV2-GPT4-valid.txt"
    tiny_set = "../data/tiny.txt"
    special_tokens = ["<|endoftext|>"]

    word_counts = parallel_pretokenize(Tiny_validation_set, special_tokens, num_processes)
    # print(word_counts)


    # first we want to remove all special tokens before pre tokenization
    # then we do pre tokenization on each of the documents. There should be one running dict[tuple[bytes], int] interface that gets updated with each document 
    # then we probably want some way to aggregate those dicts
    # END PRETOKENIZATION


    # then, using the aggregated dict as input, 
    # we figure out what merges to do and do the merging process, t

if __name__ == '__main__':
    profile = False 
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
    