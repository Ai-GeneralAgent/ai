from array import array
import hashlib
import json
from pathlib import Path
import time
from typing import Iterator, Optional, Tuple, Union

import numpy as np
import torch
from tokenizers import Tokenizer


class CompactTokenCache:
    """
    Bounded-memory tokenization and disk-based caching using 2-byte unsigned integers (u16).
    Enables instant memory-mapped random access training on multi-gigabyte corpora.
    """

    def __init__(self, cache_file: Path, token_count: int):
        self.cache_file = cache_file
        self.token_count = token_count
        self._mmap: Optional[np.memmap] = None

    @property
    def mmap(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(str(self.cache_file), dtype=np.uint16, mode="r")
        return self._mmap

    def get_window(self, start_idx: int, length: int) -> torch.Tensor:
        """Retrieves a sequence window via memory-mapping without loading the full file."""
        if start_idx + length > self.token_count:
            raise IndexError("Requested window exceeds token cache size")
        slice_np = self.mmap[start_idx : start_idx + length]
        return torch.from_numpy(slice_np.astype(np.int64))

    def sample_batch(
        self, begin: int, end: int, batch_size: int, block_size: int, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples a training batch (x, y) with target sequence offset by 1 token."""
        if end - begin <= block_size + 1:
            raise ValueError(f"Slice range ({begin}..{end}) is too small for block_size {block_size}")

        max_start = end - (block_size + 1)
        starts = np.random.randint(begin, max_start + 1, size=batch_size)

        xs = []
        ys = []
        for s in starts:
            window = self.get_window(s, block_size + 1)
            xs.append(window[:-1])
            ys.append(window[1:])

        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        return x, y

    def close(self):
        if self._mmap is not None:
            del self._mmap
            self._mmap = None

    @classmethod
    def build_or_load(
        cls,
        data_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        tokenizer: Optional[Tokenizer] = None,
        chunk_size_chars: int = 500_000,
        cache_suffix: str = ".bpe_tokens.u16",
    ) -> "CompactTokenCache":
        """
        Builds a compact token cache in bounded-memory chunks, or loads existing valid cache.
        """
        data_p = Path(data_path)
        tok_p = Path(tokenizer_path)

        if not data_p.is_file():
            raise FileNotFoundError(f"Data file not found: {data_p.resolve()}")
        if not tok_p.is_file():
            raise FileNotFoundError(f"Tokenizer not found: {tok_p.resolve()}")

        if tokenizer is None:
            tokenizer = Tokenizer.from_file(str(tok_p))

        if tokenizer.get_vocab_size() > 65536:
            raise ValueError(f"Vocab size ({tokenizer.get_vocab_size()}) exceeds uint16 capacity (65,536)")

        cache_path = data_p.with_suffix(cache_suffix)
        meta_path = Path(str(cache_path) + ".json")

        d_stat = data_p.stat()
        t_stat = tok_p.stat()
        signature = {
            "data_size": d_stat.st_size,
            "data_mtime_ns": d_stat.st_mtime_ns,
            "tokenizer_size": t_stat.st_size,
            "tokenizer_mtime_ns": t_stat.st_mtime_ns,
            "vocab_size": tokenizer.get_vocab_size(),
        }

        # Check existing cache validity
        if cache_path.is_file() and meta_path.is_file():
            try:
                saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (
                    all(saved_meta.get(k) == v for k, v in signature.items())
                    and cache_path.stat().st_size % 2 == 0
                    and saved_meta.get("token_count") == cache_path.stat().st_size // 2
                ):
                    return cls(cache_path, saved_meta["token_count"])
            except Exception:
                pass

        # Build cache with bounded memory in streaming chunks
        temp_path = Path(str(cache_path) + ".tmp")
        total_chars = 0
        total_tokens = 0
        started = time.time()

        print(f"Building token cache in chunks for: {data_p.name}")
        buffer = []
        buf_chars = 0

        try:
            with data_p.open("r", encoding="utf-8-sig", errors="replace") as src:
                with temp_path.open("wb") as out:
                    for line in src:
                        buffer.append(line)
                        buf_chars += len(line)
                        total_chars += len(line)

                        if buf_chars >= chunk_size_chars:
                            encoded = tokenizer.encode("".join(buffer)).ids
                            arr = array("H", encoded)
                            arr.tofile(out)
                            total_tokens += len(arr)
                            buffer.clear()
                            buf_chars = 0

                    if buffer:
                        encoded = tokenizer.encode("".join(buffer)).ids
                        arr = array("H", encoded)
                        arr.tofile(out)
                        total_tokens += len(arr)
                        buffer.clear()

                    out.flush()

            temp_path.replace(cache_path)
            elapsed = max(time.time() - started, 0.001)
            tps = int(total_tokens / elapsed)

            meta = dict(signature)
            meta["token_count"] = total_tokens
            meta["total_chars"] = total_chars
            meta["throughput_tokens_per_sec"] = tps
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            print(f"Token cache complete: {total_tokens:,} tokens ({tps:,} tokens/sec).")
            return cls(cache_path, total_tokens)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

