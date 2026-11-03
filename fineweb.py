"""Prepara FineWeb-Edu sample-10BT em shards, seguindo build-nanogpt.

O primeiro shard é validação e os seguintes são treino. O preparo é sequencial
por streaming para manter o script simples e evitar multiprocessing no Windows.
"""

import argparse
from itertools import islice
from pathlib import Path

import numpy as np

from train_gpt2 import get_encoding


def write_shards(documents, output, shard_size, encoding):
    """Insere EOS por documento e grava shards uint16 de até shard_size IDs."""
    if shard_size <= 0:
        raise ValueError("shard_size deve ser positivo")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use uma pasta vazia")
    output.mkdir(parents=True, exist_ok=True)
    buffer = np.empty(shard_size, dtype=np.uint16)
    used, index, total = 0, 0, 0

    def save(count, shard_index):
        split = "val" if shard_index == 0 else "train"
        np.save(output / f"edufineweb_{split}_{shard_index:06d}.npy", buffer[:count])

    for text in documents:
        tokens = [encoding.eot_token] + encoding.encode_ordinary(text)
        if min(tokens) < 0 or max(tokens) >= 50257:
            raise ValueError("Token fora do vocabulário GPT-2")
        offset = 0
        # Um documento pode atravessar vários shards; não se perde a sobra.
        while offset < len(tokens):
            amount = min(shard_size - used, len(tokens) - offset)
            buffer[used:used + amount] = tokens[offset:offset + amount]
            used += amount
            offset += amount
            total += amount
            if used == shard_size:
                save(used, index)
                index += 1
                used = 0
    if used:
        save(used, index)
        index += 1
    if index < 2:
        raise ValueError("A amostra não produziu treino e validação; use mais documentos ou shards menores")
    return total, index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/edu_fineweb10B")
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--max-documents", type=int)
    args = parser.parse_args(argv)
    if args.shard_size <= 0 or (args.max_documents is not None and args.max_documents <= 0):
        parser.error("shard-size e max-documents devem ser positivos")
    if Path(args.output).exists() and any(Path(args.output).iterdir()):
        parser.error("Use uma pasta de saída vazia")
    from datasets import load_dataset

    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    documents = (row["text"] for row in dataset)
    if args.max_documents is not None:
        documents = islice(documents, args.max_documents)
    total, count = write_shards(documents, args.output, args.shard_size, get_encoding())
    print(f"{total} tokens em {count} shards; primeiro shard reservado para validação")


if __name__ == "__main__":
    main()
