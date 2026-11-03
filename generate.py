"""Gera texto com nossa arquitetura, usando checkpoint ou pesos GPT-2 oficiais."""

import argparse

import torch

from train_gpt2 import GPT, choose_device, generate, get_encoding, load_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--pretrained", action="store_true")
    parser.add_argument("--prompt", default="Hello, I'm a language model,")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model = GPT.from_pretrained().to(device) if args.pretrained else load_checkpoint(args.checkpoint, device)[0]
    encoding = get_encoding()
    ids = encoding.encode_ordinary(args.prompt) or [encoding.eot_token]
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    result = generate(model, tokens, args.max_new_tokens, args.temperature, args.top_k, rng)
    print(encoding.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
