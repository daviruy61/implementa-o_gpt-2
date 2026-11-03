"""GPT-2: arquitetura, dados e treinamento.

Referência: https://github.com/karpathy/build-nanogpt, de Andrej Karpathy.
Inclui um modelo reduzido para CPU e atenção manual para estudo.
"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


@dataclass(frozen=True)
class GPTConfig:
    """Padrão: GPT-2 Small, 124.439.808 parâmetros únicos com weight tying.

    block_size é o contexto máximo, vocab_size é o número de IDs possíveis,
    n_embd é a largura residual e n_embd / n_head é a largura de cada cabeça.
    attention_backend seleciona duas implementações da mesma operação.
    """

    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    attention_backend: str = "sdpa"

    def __post_init__(self) -> None:
        for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} deve ser um inteiro positivo")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd deve ser divisível por n_head")
        if self.attention_backend not in {"manual", "sdpa"}:
            raise ValueError("attention_backend deve ser manual ou sdpa")

    @classmethod
    def tiny(cls, vocab_size: int = 50257, **overrides) -> "GPTConfig":
        """Modelo de teste; não representa um treinamento do GPT-2 124M."""
        values = dict(block_size=64, vocab_size=vocab_size, n_layer=2,
                      n_head=4, n_embd=64)
        values.update(overrides)
        return cls(**values)


class CausalSelfAttention(nn.Module):
    """Recebe e devolve [B,T,C]; cada posição só consulta seu prefixo."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.backend = config.attention_backend
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        # Uma projeção produz Q, K e V; cada um tem formato [B,T,C].
        query, key, value = self.c_attn(x).chunk(3, dim=-1)
        query, key, value = [
            t.reshape(batch, length, self.n_head, self.head_dim).transpose(1, 2)
            for t in (query, key, value)
        ]  # [B,H,T,D], com C = H*D; cabeças e exemplos são independentes.
        if self.backend == "sdpa":
            # O PyTorch escolhe o kernel compatível; SDPA não garante Flash
            # Attention em todo hardware. dropout_p=0 mantém as duas vias iguais.
            attended = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=True
            )
        else:
            # QK^T / sqrt(D): [B,H,T,D] @ [B,H,D,T] -> [B,H,T,T].
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            allowed = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
            scores = scores.masked_fill(~allowed, float("-inf"))
            # Softmax na dimensão das chaves: cada linha soma 1; futuro recebe 0.
            probabilities = F.softmax(scores.float(), dim=-1).to(value.dtype)
            attended = probabilities @ value  # soma ponderada -> [B,H,T,D]
        # Concatenar as cabeças restaura C; a projeção mistura seus canais.
        merged = attended.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.c_proj(merged)


class MLP(nn.Module):
    """Transformação não linear por posição: C -> 4C -> C, sem misturar tokens."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        # Aproximação tanh da GELU usada no GPT-2 e na referência.
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    """Dois ramos residuais pre-norm; entrada e saída [B,T,C]."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        # LayerNorm normaliza os C canais de cada token, não o lote ou o tempo:
        # gamma * (x - media) / sqrt(variancia + eps) + beta.
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=1e-5)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # O caminho identidade preserva informação e uma via direta de gradiente.
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """GPT-2 explícito: IDs int64 [B,T] -> logits float [B,T,V] e perda opcional.

    targets já devem estar deslocados em um token pelo carregador; o forward
    não desloca novamente. Logits são escores, não probabilidades.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            "ln_f": nn.LayerNorm(config.n_embd, eps=1e-5),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._initialize)
        # Uma única matriz aprendida codifica tokens e projeta para o vocabulário.
        self.lm_head.weight = self.transformer.wte.weight
        # Há 2L adições residuais; reduzir o desvio limita o crescimento da
        # variância ao longo da profundidade, como na referência de treinamento.
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def num_parameters(self) -> int:
        """Conta parâmetros únicos: a matriz compartilhada aparece uma vez."""
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor | None = None):
        if tokens.ndim != 2 or tokens.dtype != torch.long:
            raise ValueError("tokens deve ser int64 com formato [B,T]")
        batch, length = tokens.shape
        if batch == 0 or not 0 < length <= self.config.block_size:
            raise ValueError("Lote vazio ou comprimento fora do contexto do modelo")
        if targets is not None and (targets.shape != tokens.shape or targets.dtype != torch.long):
            raise ValueError("targets deve ser int64 com o mesmo formato de tokens")
        positions = torch.arange(length, device=tokens.device)
        # [B,T,C] + [T,C]: broadcasting usa as mesmas posições em cada exemplo.
        hidden = self.transformer.wte(tokens) + self.transformer.wpe(positions)
        for block in self.transformer.h:
            hidden = block(hidden)
        logits = self.lm_head(self.transformer.ln_f(hidden))
        loss = None
        if targets is not None:
            # CE combina log-softmax e -log p(alvo) de forma numericamente estável.
            # Não aplicar softmax antes; BT posições são exemplos da classificação.
            loss = F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                                   targets.reshape(-1))
        return logits, loss

    def configure_optimizer(self, learning_rate: float, weight_decay: float,
                            device_type: str, fused: bool = True) -> torch.optim.AdamW:
        """AdamW: decay nas matrizes; biases e LayerNorm ficam sem decay."""
        parameters = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": weight_decay},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ]
        options = {}
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            options["fused"] = fused and device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=(0.9, 0.95),
                                 eps=1e-8, **options)

    @torch.no_grad()
    def load_hf_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Importa pesos GPT-2 Conv1D transpostos para nossos módulos Linear."""
        source = {key: value for key, value in state.items()
                  if not key.endswith((".attn.bias", ".attn.masked_bias"))}
        target = self.state_dict()
        if source.keys() != target.keys():
            raise ValueError("Chaves do checkpoint incompatíveis com GPT-2")
        transpose = ("attn.c_attn.weight", "attn.c_proj.weight",
                     "mlp.c_fc.weight", "mlp.c_proj.weight")
        converted = {}
        for name, value in source.items():
            value = value.t() if name.endswith(transpose) else value
            if value.shape != target[name].shape:
                raise ValueError(f"Formato incompatível para {name}: {value.shape}")
            converted[name] = value
        if not torch.equal(converted["lm_head.weight"], converted["transformer.wte.weight"]):
            raise ValueError("Checkpoint GPT-2 não compartilha embedding e lm_head")
        self.load_state_dict(converted)

    @classmethod
    def from_pretrained(cls, attention_backend: str = "sdpa") -> "GPT":
        """Baixa GPT-2 Small via dependência opcional; executa nossa arquitetura."""
        from transformers import GPT2LMHeadModel

        reference = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        model = cls(GPTConfig(attention_backend=attention_backend))
        model.load_hf_state_dict(reference.state_dict())
        return model


# -----------------------------------------------------------------------------
# Dados: IDs GPT-2 -> entradas e alvos deslocados em um token.

def get_encoding():
    """BPE GPT-2 original. A primeira utilização pode baixar seu vocabulário."""
    import tiktoken

    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(".cache/tiktoken").resolve()))
    return tiktoken.get_encoding("gpt2")


def load_tokens(filename):
    """Converte o shard uint16 para torch.long, como na errata da referência."""
    tokens = np.load(filename, allow_pickle=False)
    if tokens.ndim != 1 or tokens.dtype != np.uint16:
        raise ValueError(f"Shard inválido: {filename}; esperado vetor uint16")
    if len(tokens) and tokens.max() >= 50257:
        raise ValueError(f"ID fora do vocabulário GPT-2 em {filename}")
    return torch.tensor(tokens.astype(np.int32), dtype=torch.long)


class DataLoaderLite:
    """Lê sequencialmente shards de um split, seguindo o carregador de Karpathy.

    B é o número de sequências por processo e T é seu comprimento. Cada rank lê
    uma faixa diferente; todos avançam B*T*num_processes tokens a cada microbatch.
    A cauda que não completa um lote global é descartada ao trocar de shard.
    """

    def __init__(self, B, T, process_rank=0, num_processes=1, split="train", data_root="data"):
        if B <= 0 or T <= 0 or num_processes <= 0 or not 0 <= process_rank < num_processes:
            raise ValueError("B, T ou configuração de processos inválidos")
        if split not in {"train", "val"}:
            raise ValueError("split deve ser train ou val")
        root = Path(data_root)
        if (root / "manifest.json").exists():
            raise ValueError("Dados da versão anterior: prepare novamente em outra pasta com prepare_data.py")
        self.B, self.T = B, T
        self.process_rank, self.num_processes = process_rank, num_processes
        self.stride = B * T * num_processes
        self.shards = []
        self.shard_lengths = []
        for path in sorted(root.glob(f"*{split}*.npy")):
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.ndim != 1 or array.dtype != np.uint16:
                raise ValueError(f"Formato inválido: {path}")
            length = len(array)
            del array
            if length >= self.stride + 1:
                self.shards.append(path)
                self.shard_lengths.append(length)
            elif process_rank == 0:
                warnings.warn(f"Ignorando {path.name}: menos de {self.stride + 1} tokens")
        if not self.shards:
            raise ValueError(f"Split {split} sem shards com pelo menos {self.stride + 1} tokens; reduza B/T")
        self.reset()

    @property
    def batches_per_epoch(self):
        return sum((length - 1) // self.stride for length in self.shard_lengths)

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[0])
        self.current_position = self.B * self.T * self.process_rank

    def state_dict(self):
        # Cursor relativo ao início do lote global: serve para qualquer rank.
        return {"shard": self.current_shard,
                "position": self.current_position - self.B * self.T * self.process_rank}

    def load_state_dict(self, state):
        shard, position = state["shard"], state["position"]
        if not 0 <= shard < len(self.shards) or position < 0 or position % self.stride:
            raise ValueError("Cursor do carregador inválido")
        if position > self.shard_lengths[shard]:
            raise ValueError("Cursor além do shard")
        self.current_shard = shard
        self.tokens = load_tokens(self.shards[shard])
        self.current_position = position + self.B * self.T * self.process_rank

    def next_batch(self):
        global_position = self.current_position - self.B * self.T * self.process_rank
        if global_position + self.stride + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        buffer = self.tokens[self.current_position:self.current_position + self.B * self.T + 1]
        # Para tokens [a,b,c,d], x=[a,b,c] e y=[b,c,d]. O modelo não desloca de novo.
        x = buffer[:-1].view(self.B, self.T)
        y = buffer[1:].view(self.B, self.T)
        self.current_position += self.stride
        return x, y


# -----------------------------------------------------------------------------
# Geração e validação.

@torch.no_grad()
def generate(model, tokens, max_new_tokens, temperature=1.0, top_k=50, generator=None):
    """Geração autoregressiva com top-k + multinomial, como na referência.

    tokens é int64 [B,T]; retorna o prompt seguido dos novos IDs. Temperatura
    zero seleciona argmax. O contexto é recortado ao limite de posições do modelo.
    """
    if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0 or top_k <= 0:
        raise ValueError("Número de tokens, temperatura ou top_k inválidos")
    if tokens.ndim != 2 or tokens.dtype != torch.long or min(tokens.shape) == 0:
        raise ValueError("Prompt deve ser int64 [B,T] não vazio")
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(tokens[:, -model.config.block_size:])
            # Exclui linhas de padding do vocabulário, caso a matriz tenha 50304.
            logits = logits[:, -1, :min(50257, model.config.vocab_size)].float()
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = F.softmax(logits / temperature, dim=-1)
                top_probs, top_indices = torch.topk(probabilities, min(top_k, probabilities.size(-1)), dim=-1)
                # multinomial aceita pesos não normalizados; top-k restringe o suporte.
                selected = torch.multinomial(top_probs, 1, generator=generator)
                next_token = torch.gather(top_indices, -1, selected)
            tokens = torch.cat((tokens, next_token), dim=1)
    finally:
        model.train(was_training)
    return tokens


def autocast_context(device, use_bf16):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_bf16 else nullcontext()


@torch.no_grad()
def evaluate(model, loader, batches, device, use_bf16=False):
    """Média da CE em validação; preserva modo do modelo e cursor do carregador."""
    if batches <= 0:
        raise ValueError("batches deve ser positivo")
    previous_state, was_training = loader.state_dict(), model.training
    model.eval()
    loader.reset()
    count = min(batches, loader.batches_per_epoch)
    loss_sum = torch.zeros((), device=device)
    try:
        for _ in range(count):
            x, y = (t.to(device) for t in loader.next_batch())
            with autocast_context(device, use_bf16):
                _, loss = model(x, y)
            loss_sum += loss
        if dist.is_initialized():
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        return (loss_sum / (count * loader.num_processes)).item()
    finally:
        model.train(was_training)
        loader.load_state_dict(previous_state)


def get_lr(step, max_steps, warmup_steps, max_lr=6e-4, min_lr=6e-5):
    """Warmup linear e decaimento cosseno com as fronteiras de build-nanogpt."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def choose_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu, cuda ou cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA indisponível nesta instalação")
        torch.cuda.set_device(device)
    return device


def load_checkpoint(path, device="cpu"):
    """Restaura nosso modelo; checkpoints guardam dicionários e tensores."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "build-gpt2-study-v1":
        raise ValueError("Checkpoint da versão anterior/incompatível; use um novo treino")
    model = GPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def parse_args(argv=None):
    """Opções explícitas para executar o mesmo laço em CPU ou CUDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Pasta com train_*.npy e val_*.npy")
    parser.add_argument("--output", default="runs/gpt2")
    parser.add_argument("--preset", choices=("gpt2", "tiny"), default="gpt2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    parser.add_argument("--min-lr", type=float, default=6e-5)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "bf16"), default="auto")
    parser.add_argument("--attention", choices=("manual", "sdpa"), default="sdpa")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--pad-vocab", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--resume", help="Checkpoint salvo por esta versão")
    parser.add_argument("--stop-after", type=int, help="Parada planejada sem mudar max-steps do scheduler")
    return parser.parse_args(argv)


def train(args):
    """Executa o laço de treinamento; encerra DDP mesmo se ocorrer um erro."""
    for name in ("batch_size", "sequence_length", "max_steps", "grad_accum_steps",
                 "eval_every", "eval_batches", "save_every", "threads"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} deve ser positivo")
    if not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("warmup-steps deve estar entre 0 e max-steps-1")
    if not all(math.isfinite(x) for x in (args.learning_rate, args.min_lr, args.weight_decay, args.grad_clip)):
        raise ValueError("Hiperparâmetros devem ser finitos")
    if not 0 <= args.min_lr <= args.learning_rate or args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("Taxas, weight-decay ou grad-clip inválidos")
    if args.stop_after is not None and not 0 < args.stop_after <= args.max_steps:
        raise ValueError("stop-after deve estar entre 1 e max-steps")
    if args.pretrained and (args.preset != "gpt2" or args.pad_vocab or args.resume):
        raise ValueError("pretrained exige gpt2 sem pad-vocab e sem resume")
    torch.set_num_threads(args.threads)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = choose_device(args.device)
    if world_size > 1:
        if device.type == "cuda":
            device = choose_device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
            if not dist.is_nccl_available():
                raise RuntimeError("DDP CUDA exige backend NCCL (Linux/WSL2)")
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        return train_loop(args, device, rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def train_loop(args, device, rank, world_size):
    """Uma atualização = grad_accum_steps microbatches, com gradientes médios."""
    output = Path(args.output)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise ValueError("Pasta de saída ocupada; use outra pasta ou --resume")
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    use_bf16 = args.precision == "bf16" or (args.precision == "auto" and device.type == "cuda"
                                           and torch.cuda.is_bf16_supported())
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU sem suporte a BF16; use --precision fp32")
    # 'high' permite TF32 em GPUs compatíveis; não muda os pesos armazenados.
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    config_options = {"vocab_size": 50304 if args.pad_vocab else 50257,
                      "attention_backend": args.attention}
    config = GPTConfig.tiny(**config_options) if args.preset == "tiny" else GPTConfig(**config_options)
    if args.sequence_length > config.block_size:
        raise ValueError("sequence-length excede o contexto do modelo")
    train_loader = DataLoaderLite(args.batch_size, args.sequence_length, rank, world_size, "train", args.data)
    val_loader = DataLoaderLite(args.batch_size, args.sequence_length, rank, world_size, "val", args.data)
    data_signature = {split: [(p.name, n) for p, n in zip(loader.shards, loader.shard_lengths)]
                      for split, loader in (("train", train_loader), ("val", val_loader))}
    start_step = 0
    checkpoint = None
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device)
        if checkpoint["config"] != asdict(config) or checkpoint["world_size"] != world_size:
            raise ValueError("Retomada exige mesma arquitetura e world_size")
        if checkpoint["data"] != data_signature:
            raise ValueError("Shards diferentes dos usados no checkpoint")
        for key in ("batch_size", "sequence_length", "grad_accum_steps", "max_steps", "warmup_steps",
                    "learning_rate", "min_lr", "weight_decay", "grad_clip", "seed"):
            if checkpoint["args"][key] != getattr(args, key):
                raise ValueError(f"Retomada exige o mesmo {key}")
        if checkpoint["use_bf16"] != use_bf16:
            raise ValueError("Retomada exige a mesma precisão")
        start_step = checkpoint["step"]
        train_loader.load_state_dict(checkpoint["loader"])
    else:
        model = GPT.from_pretrained(args.attention) if args.pretrained else GPT(config)
        model.to(device)
    optimizer = model.configure_optimizer(args.learning_rate, args.weight_decay, device.type)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    raw_model = model  # Salvar os nomes originais, sem prefixos DDP/compile.
    compiled = False
    if args.compile:
        if device.type != "cuda" or sys.platform == "win32":
            warnings.warn("torch.compile desativado neste ambiente; use CUDA/Linux para essa opção")
        else:
            model = torch.compile(model)
            compiled = True
    if world_size > 1:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    if rank == 0:
        print(json.dumps({"parameters": raw_model.num_parameters(), "device": str(device),
                          "bf16": use_bf16, "compiled": compiled, "world_size": world_size,
                          "tokens_per_step": args.batch_size * args.sequence_length * args.grad_accum_steps * world_size}), flush=True)
    stop = args.max_steps if args.stop_after is None else args.stop_after
    if start_step >= stop:
        raise ValueError("Checkpoint já atingiu a etapa solicitada")
    model.train()
    for step in range(start_step, stop):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = torch.zeros((), device=device)
        for microstep in range(args.grad_accum_steps):
            x, y = (t.to(device) for t in train_loader.next_batch())
            # A sincronização deve ser configurada ANTES do forward (errata).
            if world_size > 1:
                model.require_backward_grad_sync = microstep == args.grad_accum_steps - 1
            with autocast_context(device, use_bf16):
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if world_size > 1:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)
            loss_accum /= world_size
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if not torch.isfinite(loss_accum) or not torch.isfinite(norm):
            raise FloatingPointError("Perda ou gradientes não finitos")
        lr = get_lr(step, args.max_steps, args.warmup_steps, args.learning_rate, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {"step": step + 1, "train_loss": loss_accum.item(), "lr": lr,
                  "grad_norm": norm.item(), "seconds": time.perf_counter() - started}
        if (step + 1) % args.eval_every == 0 or step + 1 == stop:
            record["val_loss"] = evaluate(model, val_loader, args.eval_batches, device, use_bf16)
        if rank == 0:
            print(json.dumps(record), flush=True)
            with (output / "log.txt").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        if (step + 1) % args.save_every == 0 or step + 1 == stop:
            if rank == 0:
                # Não há dropout, embaralhamento ou amostragem no laço: depois da
                # inicialização, o treino não consome RNG. Cursor e otimizador são
                # suficientes para retomar este fluxo determinístico em CPU.
                state = {"format": "build-gpt2-study-v1", "model": raw_model.state_dict(),
                         "config": asdict(config), "optimizer": optimizer.state_dict(),
                         "step": step + 1, "loader": train_loader.state_dict(),
                         "args": vars(args), "world_size": world_size,
                         "data": data_signature, "use_bf16": use_bf16}
                torch.save(state, output / "last.pt.tmp")
                os.replace(output / "last.pt.tmp", output / "last.pt")
            if world_size > 1:
                dist.barrier()
    return record


if __name__ == "__main__":
    train(parse_args())
