# import multiprocessing as mp 
from pprint import pprint
import regex as re
from collections import defaultdict

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pass

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
            word_counts[match.group()] += 1
    return word_counts

def main():
    special_tokens = ["<|endoftext|>"]
    Tiny_train_set = "../data/TinyStoriesV2-GPT4-train.txt"
    Tiny_validation_set = "../data/TinyStoriesV2-GPT4-valid.txt"
    tiny_set = "../data/tiny.txt"
    with open(Tiny_validation_set, 'r') as f:
        word_counts = pretokenize(f.read(), special_tokens)

    # first we want to remove all special tokens before pre tokenization
    # then we do pre tokenization on each of the documents. There should be one running dict[tuple[bytes], int] interface that gets updated with each document 
    # then we probably want some way to aggregate those dicts
    # END PRETOKENIZATION


    # then, using the aggregated dict as input, 
    # we figure out what merges to do and do the merging process, t

if __name__ == '__main__':
    main() 
    