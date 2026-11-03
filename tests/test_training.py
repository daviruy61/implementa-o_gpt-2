"""Treinamento reduzido, checkpoint, validação e scheduler."""

import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train_gpt2 import DataLoaderLite, GPT, GPTConfig, evaluate, get_lr, load_checkpoint, parse_args, train, train_loop


def test_scheduler_matches_reference_boundaries():
    assert get_lr(0, 10, 2) == 3e-4
    assert get_lr(1, 10, 2) == 6e-4
    assert get_lr(2, 10, 2) == 6e-4
    assert get_lr(10, 10, 2) == 6e-5
    assert get_lr(11, 10, 2) == 6e-5
    assert get_lr(9, 10, 2) > 6e-5


def test_validation_restores_state(token_data):
    model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=16, vocab_size=256))
    loader = DataLoaderLite(2, 8, split="val", data_root=token_data)
    loader.next_batch()
    before = loader.state_dict()
    loss = evaluate(model, loader, 2, torch.device("cpu"))
    assert 0 < loss < 10 and model.training and loader.state_dict() == before


def test_training_resume_and_generation(token_data, tmp_path):
    common = ["--data", str(token_data), "--preset", "tiny", "--batch-size", "1",
              "--sequence-length", "8", "--max-steps", "4", "--warmup-steps", "1",
              "--grad-accum-steps", "2", "--eval-every", "2", "--eval-batches", "1",
              "--save-every", "2", "--device", "cpu", "--threads", "2"]
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    train(parse_args(common + ["--output", str(full)]))
    train(parse_args(common + ["--output", str(resumed), "--stop-after", "2"]))
    train(parse_args(common + ["--output", str(resumed), "--resume", str(resumed / "last.pt")]))
    model_a, checkpoint_a = load_checkpoint(full / "last.pt")
    model_b, checkpoint_b = load_checkpoint(resumed / "last.pt")
    assert checkpoint_a["step"] == checkpoint_b["step"] == 4
    assert checkpoint_a["loader"] == checkpoint_b["loader"]
    for key in model_a.state_dict():
        torch.testing.assert_close(model_a.state_dict()[key], model_b.state_dict()[key], atol=0, rtol=0)
    for key, state in checkpoint_a["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, checkpoint_b["optimizer"]["state"][key][name], atol=0, rtol=0)
    assert model_b.lm_head.weight is model_b.transformer.wte.weight
    records = [json.loads(line) for line in (resumed / "log.txt").read_text().splitlines()]
    assert [record["step"] for record in records] == [1, 2, 3, 4]


def test_bf16_cpu_and_compile_fallback(token_data, tmp_path):
    args = parse_args(["--data", str(token_data), "--output", str(tmp_path / "bf16"),
                       "--preset", "tiny", "--batch-size", "1", "--sequence-length", "8",
                       "--max-steps", "2", "--warmup-steps", "0", "--eval-batches", "1",
                       "--precision", "bf16", "--compile", "--device", "cpu", "--threads", "2"])
    with pytest.warns(UserWarning, match="compile desativado"):
        result = train(args)
    assert result["step"] == 2 and result["val_loss"] > 0


def distributed_worker(rank, data, output, store_path):
    # FileStore somente no teste: permite verificar o laço DDP no Windows sem
    # adicionar um launcher alternativo ao projeto que será estudado.
    torch.set_num_threads(1)
    dist.init_process_group("gloo", store=dist.FileStore(store_path, 2), rank=rank, world_size=2)
    try:
        args = parse_args(["--data", data, "--output", output, "--preset", "tiny",
                           "--batch-size", "1", "--sequence-length", "8", "--max-steps", "2",
                           "--warmup-steps", "0", "--grad-accum-steps", "2", "--eval-batches", "1"])
        train_loop(args, torch.device("cpu"), rank, 2)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo indisponível")
def test_ddp_matches_single_process(token_data, tmp_path):
    distributed_output = tmp_path / "ddp"
    mp.spawn(distributed_worker, args=(str(token_data), str(distributed_output), str(tmp_path / "store")),
             nprocs=2, join=True)
    single_output = tmp_path / "single"
    args = parse_args(["--data", str(token_data), "--output", str(single_output), "--preset", "tiny",
                       "--batch-size", "2", "--sequence-length", "8", "--max-steps", "2",
                       "--warmup-steps", "0", "--grad-accum-steps", "2", "--eval-batches", "1",
                       "--device", "cpu", "--threads", "1"])
    train(args)
    distributed_model, state = load_checkpoint(distributed_output / "last.pt")
    single_model, _ = load_checkpoint(single_output / "last.pt")
    assert state["step"] == 2 and state["world_size"] == 2
    with torch.no_grad():
        tokens = torch.tensor([[1, 2, 3, 4]])
        actual, _ = distributed_model(tokens)
        expected, _ = single_model(tokens)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)
