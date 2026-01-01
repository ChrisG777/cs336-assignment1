import os
import codecs
import multiprocessing as mp
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

# Use your existing helper
from .pretokenization_example import find_chunk_boundaries
from .tokenizer import Tokenizer  # adjust import path to your Tokenizer


# -----------------------------
# Fast streaming writer (piece-level, not token-yield-level)
# -----------------------------
def _encode_iterable_to_rawbin(
    tok: Tokenizer,
    iterable: Iterable[str],
    out_fh,
    *,
    typecode: str,
    flush_tokens: int = 1_000_000,
) -> int:
    """
    Tokenize a stream of text chunks and write token IDs to out_fh as raw array(typecode).

    This is like tokenizer.encode_iterable(), but avoids per-token Python generator overhead by:
      - streaming special-token splitting
      - streaming regex pretokenization
      - calling tok.encode_word(piece) and array.extend(list_of_ids)
    """
    buf = array(typecode)
    total = 0

    # Hot references
    pat_finditer = tok._pat.finditer
    encode_word = tok.encode_word

    # Flush helper
    def flush() -> None:
        nonlocal buf
        if buf:
            buf.tofile(out_fh)
            buf = array(typecode)

    def emit_ids(ids: list[int]) -> None:
        nonlocal total
        if ids:
            buf.extend(ids)
            total += len(ids)
            if len(buf) >= flush_tokens:
                flush()

    def emit_one(tid: int) -> None:
        nonlocal total
        buf.append(tid)
        total += 1
        if len(buf) >= flush_tokens:
            flush()

    # -----------------------------
    # Streaming regex with a carry buffer:
    # defer the last match if it reaches end-of-buffer (when not at a hard boundary)
    # -----------------------------
    pbuf = ""

    def feed_pbuf(s: str, *, final: bool) -> None:
        nonlocal pbuf
        if not s and (not final or not pbuf):
            return
        pbuf = pbuf + s
        if not pbuf:
            return

        last_start = None
        for m in pat_finditer(pbuf):
            end = m.end()
            if (not final) and end == len(pbuf):
                last_start = m.start()
                break
            emit_ids(encode_word(m.group(0)))

        if last_start is None:
            pbuf = ""
        else:
            pbuf = pbuf[last_start:]

    # -----------------------------
    # No special tokens: just regex stream
    # -----------------------------
    if not tok.special_tokens:
        for chunk in iterable:
            if chunk:
                feed_pbuf(chunk, final=False)
        feed_pbuf("", final=True)
        flush()
        return total

    # -----------------------------
    # With special tokens: scan specials across chunk boundaries
    # -----------------------------
    sbuf = ""
    max_keep = max(0, tok._max_special_len - 1)

    # Single special token fast path
    if tok._single_special is not None:
        special = tok._single_special
        sid = tok._single_special_id

        for chunk in iterable:
            if not chunk:
                continue
            sbuf += chunk

            while True:
                idx = sbuf.find(special)
                if idx == -1:
                    break

                # hard boundary: flush regex up to idx as final
                if idx:
                    feed_pbuf(sbuf[:idx], final=True)
                # also flush any deferred tail
                feed_pbuf("", final=True)

                emit_one(sid)
                sbuf = sbuf[idx + len(special) :]

            # keep a suffix that might start a special token
            if max_keep and len(sbuf) > max_keep:
                cut = len(sbuf) - max_keep
                feed_pbuf(sbuf[:cut], final=False)
                sbuf = sbuf[cut:]

        # end-of-stream flush
        if sbuf:
            feed_pbuf(sbuf, final=True)
        feed_pbuf("", final=True)
        flush()
        return total

    # Multiple specials
    special_re = tok._special_re
    special_to_id = tok._special_to_id

    for chunk in iterable:
        if not chunk:
            continue
        sbuf += chunk

        while True:
            m = special_re.search(sbuf)
            if m is None:
                break

            a, b = m.span()
            if a:
                feed_pbuf(sbuf[:a], final=True)
            feed_pbuf("", final=True)

            emit_one(special_to_id[m.group(0)])
            sbuf = sbuf[b:]

        if max_keep and len(sbuf) > max_keep:
            cut = len(sbuf) - max_keep
            feed_pbuf(sbuf[:cut], final=False)
            sbuf = sbuf[cut:]

    if sbuf:
        feed_pbuf(sbuf, final=True)
    feed_pbuf("", final=True)
    flush()
    return total


def _iter_text_from_file_range(
    path: str,
    start: int,
    end: int,
    *,
    block_bytes: int,
) -> Iterator[str]:
    """
    Stream decode UTF-8 from a byte range without needing the whole chunk in RAM.
    Boundaries are expected to be at safe locations (we'll make them land on ASCII '<' of <|endoftext|>).
    """
    dec = codecs.getincrementaldecoder("utf-8")()
    remaining = end - start
    with open(path, "rb") as f:
        f.seek(start)
        while remaining > 0:
            b = f.read(min(block_bytes, remaining))
            if not b:
                break
            remaining -= len(b)
            s = dec.decode(b)
            if s:
                yield s
        tail = dec.decode(b"", final=True)
        if tail:
            yield tail


# -----------------------------
# Worker globals (avoid pickling tokenizer)
# -----------------------------
_G_TOK = None
_G_TYPECODE = None
_G_BLOCK_BYTES = None
_G_FLUSH_TOKENS = None


def _worker_init(
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    typecode: str,
    block_bytes: int,
    flush_tokens: int,
) -> None:
    global _G_TOK, _G_TYPECODE, _G_BLOCK_BYTES, _G_FLUSH_TOKENS
    _G_TOK = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    _G_TYPECODE = typecode
    _G_BLOCK_BYTES = block_bytes
    _G_FLUSH_TOKENS = flush_tokens


def _encode_chunk_to_shard(task: tuple[int, str, int, int, str]) -> tuple[int, int, str]:
    """
    Returns (chunk_index, token_count, shard_path)
    """
    idx, input_path, start, end, shard_path = task
    tok = _G_TOK
    typecode = _G_TYPECODE
    block_bytes = _G_BLOCK_BYTES
    flush_tokens = _G_FLUSH_TOKENS

    Path(shard_path).parent.mkdir(parents=True, exist_ok=True)

    with open(shard_path, "wb") as out_f:
        n_tokens = _encode_iterable_to_rawbin(
            tok,
            _iter_text_from_file_range(input_path, start, end, block_bytes=block_bytes),
            out_f,
            typecode=typecode,
            flush_tokens=flush_tokens,
        )
    return idx, n_tokens, shard_path


# -----------------------------
# Main: parallel encode -> one .npy
# -----------------------------
def parallel_encode_to_npy(
    *,
    input_path: str,
    vocab_path: str,
    merges_path: str,
    output_npy_path: str,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_workers: int = 48,
    num_chunks: int | None = None,
    tmp_dir: str = "./_tok_shards_tmp",
    block_bytes: int = 8 << 20,          # 8MB
    flush_tokens: int = 1_000_000,       # write every ~1M tokens
    cleanup_shards: bool = True,
) -> dict:
    """
    Produces a SINGLE .npy with tokens in EXACT input order.

    Strategy:
      1) split the file into `num_chunks` byte ranges aligned to the start of a special token
      2) encode each range -> raw shard .bin (uint16/uint32)
      3) concatenate shards into one .npy (header + raw data) in increasing chunk index order
    """
    if num_chunks is None:
        # more chunks than workers improves load-balancing
        num_chunks = max(num_workers * 4, num_workers)

    # Choose dtype once (assumes vocab ids are <= max(vocab))
    tok0 = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    max_id = max(tok0.vocab) if tok0.vocab else 0
    if max_id <= 65535:
        dtype = np.uint16
        typecode = "H"
        itemsize = 2
    else:
        dtype = np.uint32
        typecode = "I"
        itemsize = 4

    # Delimiter for chunking: for OWT it’s <|endoftext|>
    if not special_tokens:
        raise ValueError("You must pass special_tokens; for OWT use ['<|endoftext|>'].")
    delimiter = special_tokens[0].encode("utf-8")

    # Compute aligned boundaries
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, delimiter)

    tasks: list[tuple[int, str, int, int, str]] = []
    tmp_dir_p = Path(tmp_dir)
    tmp_dir_p.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        shard_path = str(tmp_dir_p / f"shard_{chunk_idx:06d}.bin")
        tasks.append((chunk_idx, input_path, start, end, shard_path))
        chunk_idx += 1

    # Use fork on Linux for speed + memory sharing
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()

    with ctx.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(vocab_path, merges_path, special_tokens, typecode, block_bytes, flush_tokens),
    ) as pool:
        results = list(pool.imap_unordered(_encode_chunk_to_shard, tasks, chunksize=1))

    # Put results in chunk order
    results.sort(key=lambda x: x[0])
    shard_paths = [p for _, _, p in results]
    shard_token_counts = [n for _, n, _ in results]

    total_tokens = sum(shard_token_counts)

    # Create final .npy (correct header) and then fill data region by concatenating raw shard bytes
    out_path = Path(output_npy_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.lib.format.open_memmap(
        output_npy_path,
        mode="w+",
        dtype=dtype,
        shape=(total_tokens,),
    )
    data_offset = arr.offset
    del arr  # close memmap

    # Copy each shard into the .npy's data section
    # Use buffered I/O for cross-platform compatibility (sendfile differs on macOS vs Linux)
    COPY_BUFSIZE = 8 * 1024 * 1024  # 8MB buffer

    with open(output_npy_path, "r+b") as out_f:
        out_f.seek(data_offset)

        for sp in shard_paths:
            shard_size = os.path.getsize(sp)
            if shard_size % itemsize != 0:
                raise ValueError(f"Shard {sp} has size {shard_size} not divisible by itemsize {itemsize}.")

            with open(sp, "rb") as in_f:
                while True:
                    chunk = in_f.read(COPY_BUFSIZE)
                    if not chunk:
                        break
                    out_f.write(chunk)

    if cleanup_shards:
        for sp in shard_paths:
            try:
                os.remove(sp)
            except OSError:
                pass

    return {
        "output_npy_path": str(out_path),
        "dtype": str(np.dtype(dtype)),
        "num_chunks": len(shard_paths),
        "num_workers": num_workers,
        "total_tokens": int(total_tokens),
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Parallel tokenization of a text file to .npy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="./data/TinyStoriesV2-GPT4-train.txt",
                        help="Input text file to tokenize")
    parser.add_argument("--vocab", type=str, default="vocab.json",
                        help="Path to vocabulary JSON file")
    parser.add_argument("--merges", type=str, default="merges.txt",
                        help="Path to merges file")
    parser.add_argument("--output", type=str, default="tokens.npy",
                        help="Output .npy file path")
    parser.add_argument("--special-tokens", type=str, nargs="*", default=["<|endoftext|>"],
                        help="Special tokens")
    parser.add_argument("--num-workers", type=int, default=12,
                        help="Number of parallel workers")
    parser.add_argument("--num-chunks", type=int, default=None,
                        help="Number of chunks (default: 4x workers)")
    parser.add_argument("--tmp-dir", type=str, default="./_tok_shards_tmp",
                        help="Temporary directory for shards (use /scratch on HPC clusters)")

    args = parser.parse_args()

    stats = parallel_encode_to_npy(
        input_path=args.input,
        vocab_path=args.vocab,
        merges_path=args.merges,
        output_npy_path=args.output,
        special_tokens=args.special_tokens,
        num_workers=args.num_workers,
        num_chunks=args.num_chunks if args.num_chunks else args.num_workers * 4,
        tmp_dir=args.tmp_dir,
    )
    print(stats)
