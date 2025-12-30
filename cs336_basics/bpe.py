# import multiprocessing as mp 
from pprint import pprint
import re

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pass


def split_documents(input_path: str, special_tokens: list[str]) -> list[str]:
    special_tokens_escaped = [re.escape(token) for token in special_tokens] 
    special_pattern = "|".join(special_tokens_escaped)
    with open(input_path, 'r') as f:
        body = f.read() 
        documents = re.split(special_pattern, body)
    return [document for document in documents if document != '']

if __name__ == '__main__':
    special_tokens = ["<|endoftext|>"]
    documents = split_documents('../data/tiny.txt', special_tokens)[-1]
    

    # first we want to remove all special tokens before pre tokenization
    # then we do pre tokenization on each of the documents. There should be one running dict[tuple[bytes], int] interface that gets updated with each document 
    # then we probably want some way to aggregate those dicts
    # END PRETOKENIZATION


    # then, using the aggregated dict as input, 
    # we figure out what merges to do and do the merging process, t

    # use TinyStoriesV2-GPT4-valid.txt first
    