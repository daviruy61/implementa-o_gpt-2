"""Contratos matemáticos da arquitetura, gradientes e amostragem."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from train_gpt2 import GPT, GPTConfig, generate


def small_config(**overrides):
    values = dict(block_size=16, vocab_size=32, n_layer=2, n_head=2, n_embd=16)
    values.update(overrides)
    return GPTConfig(**values)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_forward_loss_backward_and_update(backend):
    model = GPT(small_config(attention_backend=backend))
    tokens = torch.randint(32, (2, 9))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 8, 32)
    assert loss.ndim == 0
    # Cálculo independente: -log da probabilidade do alvo em cada posição.
    expected = -F.log_softmax(logits, -1).gather(-1, targets.unsqueeze(-1)).mean()
    torch.testing.assert_close(loss, expected)
    optimizer = model.configure_optimizer(1e-3, 0.1, "cpu")
    before = model.transformer.wte.weight.detach().clone()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.transformer.wte.weight)
    assert model.lm_head.weight is model.transformer.wte.weight
    all_params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(all_params) == len({id(p) for p in all_params})
    assert all(p.ndim >= 2 for p in optimizer.param_groups[0]["params"])
    assert all(p.ndim < 2 for p in optimizer.param_groups[1]["params"])


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_future_tokens_cannot_change_prefix(backend):
    model = GPT(small_config(attention_backend=backend)).eval()
    tokens = torch.randint(32, (2, 12))
    changed = tokens.clone()
    changed[:, 5:] = (changed[:, 5:] + 7) % 32
    with torch.no_grad():
        original, _ = model(tokens)
        perturbed, _ = model(changed)
    torch.testing.assert_close(original[:, :5], perturbed[:, :5], atol=1e-6, rtol=1e-5)
    assert not torch.allclose(original[:, 5:], perturbed[:, 5:])
    # Testa também a dependência diferencial, sem o caminho do embedding compartilhado.
    hidden = torch.randn(1, 8, 16, requires_grad=True)
    model.transformer.h[0].attn(hidden)[:, :3].sum().backward()
    assert torch.count_nonzero(hidden.grad[:, 3:]) == 0


def test_sdpa_matches_manual_outputs_and_gradients():
    manual = GPT(small_config(attention_backend="manual"))
    optimized = GPT(replace(manual.config, attention_backend="sdpa"))
    optimized.load_state_dict(manual.state_dict())
    inputs, targets = torch.randint(32, (2, 7)), torch.randint(32, (2, 7))
    logits_a, loss_a = manual(inputs, targets)
    logits_b, loss_b = optimized(inputs, targets)
    torch.testing.assert_close(logits_a, logits_b, atol=1e-6, rtol=1e-5)
    loss_a.backward()
    loss_b.backward()
    for a, b in zip(manual.parameters(), optimized.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)


def test_gradient_accumulation_equals_full_batch():
    full = GPT(small_config())
    accumulated = GPT(small_config())
    accumulated.load_state_dict(full.state_dict())
    inputs, targets = torch.randint(32, (4, 8)), torch.randint(32, (4, 8))
    full(inputs, targets)[1].backward()
    for x, y in zip(inputs.chunk(2), targets.chunk(2)):
        (accumulated(x, y)[1] / 2).backward()
    for a, b in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)


def test_can_overfit_a_fixed_batch():
    model = GPT(small_config())
    inputs = torch.arange(8).repeat(2, 1)
    targets = inputs + 1
    optimizer = model.configure_optimizer(0.02, 0, "cpu")
    initial = model(inputs, targets)[1].item()
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        model(inputs, targets)[1].backward()
        optimizer.step()
    final = model(inputs, targets)[1].item()
    assert final < initial * 0.25


def test_generation_seed_context_and_mode():
    model = GPT(small_config(block_size=4))
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    a = generate(model, prompt, 7, top_k=5,
                       generator=torch.Generator().manual_seed(99))
    b = generate(model, prompt, 7, top_k=5,
                       generator=torch.Generator().manual_seed(99))
    assert model.training
    assert a.shape == (1, 12) and torch.equal(a, b)
    assert torch.equal(a[:, :5], prompt)
    assert a.min() >= 0 and a.max() < 32


def test_top_k_and_padded_vocabulary():
    model = GPT(small_config(vocab_size=50304))
    prompt = torch.tensor([[1]])
    # Mesmo que um ID artificial tenha logit maior, só amostramos IDs GPT-2.
    scores = torch.full((1, 1, 50304), -100.0)
    scores[0, 0, :3] = torch.tensor([0.1, 0.3, 0.6]).log()
    scores[0, 0, 50257] = 100
    with patch.object(model, "forward", return_value=(scores, None)):
        for options in ({"temperature": 0}, {"top_k": 1}):
            result = generate(model, prompt, 3, **options)
            assert result.tolist() == [[1, 2, 2, 2]]


def test_invalid_config_and_tensor_shapes():
    with pytest.raises(ValueError, match="divisível"):
        GPTConfig(n_embd=17, n_head=2)
    model = GPT(small_config())
    with pytest.raises(ValueError, match="int64"):
        model(torch.zeros(1, 3))
    with pytest.raises(ValueError, match="contexto"):
        model(torch.zeros(1, 17, dtype=torch.long))
    with pytest.raises(ValueError, match="targets"):
        model(torch.ones(1, 3, dtype=torch.long), torch.ones(1, 2, dtype=torch.long))


@pytest.mark.slow
def test_real_gpt2_124m_instantiation_and_forward():
    # Instancia pesos reais em CPU, não apenas tensores meta nem fórmula estimada.
    model = GPT(GPTConfig()).eval()
    assert model.num_parameters() == 124_439_808
    assert model.lm_head.weight is model.transformer.wte.weight
    with torch.no_grad():
        logits, loss = model(torch.tensor([[15496, 11, 314, 716]]), torch.tensor([[11, 314, 716, 257]]))
    assert logits.shape == (1, 4, 50257)
    assert torch.isfinite(logits).all() and torch.isfinite(loss)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
def test_optional_huggingface_numerical_parity(backend):
    transformers = pytest.importorskip("transformers")
    # Referência independente com pesos aleatórios pequenos: nenhum download.
    config = transformers.GPT2Config(vocab_size=32, n_positions=16, n_ctx=16,
                                     n_embd=16, n_layer=2, n_head=2,
                                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
                                     attn_implementation="eager")
    reference = transformers.GPT2LMHeadModel(config).eval()
    model = GPT(small_config(attention_backend=backend)).eval()
    model.load_hf_state_dict(reference.state_dict())
    tokens = torch.randint(32, (2, 9))
    with torch.no_grad():
        actual, _ = model(tokens)
        expected = reference(tokens, labels=tokens)
        _, loss = model(tokens[:, :-1], tokens[:, 1:])
    torch.testing.assert_close(actual, expected.logits, atol=2e-6, rtol=1e-4)
    torch.testing.assert_close(loss, expected.loss, atol=2e-6, rtol=1e-5)
