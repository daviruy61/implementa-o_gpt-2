"""Lotes, divisão entre ranks e preparação de shards como na referência."""

import numpy as np
import pytest
import torch

from fineweb import write_shards
import prepare_data
from train_gpt2 import DataLoaderLite


def test_targets_and_shard_transition(token_data):
    loader = DataLoaderLite(2, 8, data_root=token_data)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (2, 8) and x.dtype == torch.long
    assert x.flatten().tolist() == list(range(16))
    assert y.flatten().tolist() == list(range(1, 17))
    for _ in range(3):
        loader.next_batch()
    # Restam seis tokens no primeiro shard; a próxima leitura começa no segundo.
    x, y = loader.next_batch()
    assert loader.current_shard == 1
    assert x[0, 0].item() == 70 and y[0, 0].item() == 71


def test_ranks_and_cursor_restore(token_data):
    ranks = [DataLoaderLite(1, 8, rank, 2, data_root=token_data) for rank in range(2)]
    single = DataLoaderLite(2, 8, data_root=token_data)
    for _ in range(12):
        expected = single.next_batch()
        batches = [loader.next_batch() for loader in ranks]
        for i in range(2):
            torch.testing.assert_close(torch.cat([batch[i] for batch in batches]), expected[i])
        assert ranks[0].state_dict() == ranks[1].state_dict() == single.state_dict()
    state = single.state_dict()
    expected = single.next_batch()
    single.load_state_dict(state)
    actual = single.next_batch()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_short_shards(token_data):
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="sem shards"):
        DataLoaderLite(100, 100, data_root=token_data)


class ExampleEncoding:
    """Substituto apenas de teste; produção sempre usa tiktoken GPT-2."""
    eot_token = 50256

    def encode_ordinary(self, text):
        return [ord(letter) for letter in text]


def test_prepare_text(tmp_path, monkeypatch):
    source = tmp_path / "input.txt"
    source.write_text("abcde" * 100, encoding="utf-8")
    output = tmp_path / "prepared"
    monkeypatch.setattr(prepare_data, "get_encoding", ExampleEncoding)
    prepare_data.main(["--input", str(source), "--output", str(output)])
    train = np.load(output / "train_000000.npy")
    val = np.load(output / "val_000000.npy")
    assert len(train) == 450 and len(val) == 50
    assert np.concatenate([train, val]).tolist() == [ord(c) for c in "abcde" * 100]


def test_fineweb_first_shard_is_validation_and_preserves_tokens(tmp_path):
    output = tmp_path / "fineweb"
    texts = ["a" * 25, "bc", "d" * 15]
    encoding = ExampleEncoding()
    total, count = write_shards(texts, output, 10, encoding)
    files = sorted(output.glob("*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
    expected = [token for text in texts for token in [50256] + encoding.encode_ordinary(text)]
    assert count == 5 and total == len(expected)
    assert "val" in files[0].name and all("train" in path.name for path in files[1:])
    assert np.concatenate([np.load(path) for path in files]).tolist() == expected
