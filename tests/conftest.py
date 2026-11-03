"""IDs sintéticos nos testes: não é necessário baixar o corpus ou o BPE."""

from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch


def pytest_configure(config):
    """Isola temporários por execução, sem depender do pytest-of-usuario global."""
    if config.option.basetemp is None:
        # No Windows, execuções do usuário e do sandbox podem compartilhar TEMP,
        # mas criar pastas privadas com proprietários diferentes. Cada execução
        # cria sua própria pasta e a remove ao encerrar o pytest. Mantê-la no
        # TEMP também evita o caminho com acentos do projeto no FileStore/Gloo.
        temporary = TemporaryDirectory(prefix="gpt2-tests-")
        config.option.basetemp = temporary.name
        config.add_cleanup(temporary.cleanup)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(1234)


@pytest.fixture
def token_data(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    np.save(root / "train_000000.npy", np.arange(70, dtype=np.uint16))
    np.save(root / "train_000001.npy", np.arange(70, 160, dtype=np.uint16))
    np.save(root / "val_000000.npy", np.arange(128, dtype=np.uint16))
    return root
