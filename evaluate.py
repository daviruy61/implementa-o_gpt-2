"""Calcula perda de validação e perplexidade de um checkpoint local."""

import argparse
import json
import math

import torch

from train_gpt2 import DataLoaderLite, choose_device, evaluate, load_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    loader = DataLoaderLite(args.batch_size, args.sequence_length, split="val", data_root=args.data)
    loss = evaluate(model, loader, args.batches, device)
    print(json.dumps({"val_loss": loss, "perplexity": math.exp(loss),
                      "batches": min(args.batches, loader.batches_per_epoch)}))


if __name__ == "__main__":
    main()
