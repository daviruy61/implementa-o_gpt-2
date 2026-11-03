"""Prepara um texto UTF-8 com o tokenizador GPT-2 para o treino inicial."""

import argparse
from pathlib import Path

import numpy as np

from train_gpt2 import get_encoding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args(argv)
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction deve estar em (0,1)")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Use uma pasta de saída vazia")
    text = args.input.read_text(encoding="utf-8")
    tokens = np.array(get_encoding().encode_ordinary(text), dtype=np.uint16)
    boundary = int(len(tokens) * (1 - args.val_fraction))
    if min(boundary, len(tokens) - boundary) < 2:
        parser.error("Texto insuficiente para separar treino e validação")
    args.output.mkdir(parents=True, exist_ok=True)
    # Tiny Shakespeare é usado inicialmente na referência. Aqui o mesmo caminho
    # aceita qualquer texto local, e reserva o trecho final para validação.
    np.save(args.output / "train_000000.npy", tokens[:boundary])
    np.save(args.output / "val_000000.npy", tokens[boundary:])
    print(f"Treino: {boundary} tokens | validação: {len(tokens) - boundary} tokens | BPE GPT-2")


if __name__ == "__main__":
    main()
