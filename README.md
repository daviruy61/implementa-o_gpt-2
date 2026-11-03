# Implementação do GPT-2

Projeto de estudo baseado em [build-nanogpt, de Andrej Karpathy](https://github.com/karpathy/build-nanogpt).
A arquitetura e o treinamento estão em `train_gpt2.py`.

## 1. Instalar

Use Python 3.12 e abra o PowerShell na pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se o ambiente já existe, basta o segundo comando. Não é necessário ativar a `.venv`.
Todas as dependências estão no mesmo arquivo, incluindo Transformers e datasets.

## 2. Executar os testes

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes não baixam corpus nem pesos. Os resultados estão em [VALIDACAO.md](VALIDACAO.md).

## 3. Preparar o texto

```powershell
.\.venv\Scripts\python.exe prepare_data.py --input examples/corpus.txt --output data/estudo
```

O script usa o tokenizador GPT-2 e separa 90% dos tokens para treino e 10% para validação.
A primeira execução pode baixar o vocabulário. Para usar outro texto, troque o caminho de `--input`.
Se `data/estudo` já foi preparada nesta versão, pule esta etapa ou escolha outra pasta.

## 4. Treinar um modelo pequeno em CPU

```powershell
.\.venv\Scripts\python.exe train_gpt2.py --data data/estudo --output runs/estudo `
  --preset tiny --batch-size 1 --sequence-length 8 `
  --max-steps 10 --warmup-steps 2 --device cpu
```

Esse comando faz um teste rápido com o modelo `tiny`, de aproximadamente 3,3 milhões de parâmetros.
Ele salva os pesos em `runs/estudo/last.pt` e as métricas em `runs/estudo/log.txt`.
Um treino tão curto serve para verificar o funcionamento; o texto gerado ainda será incoerente.

Se `runs/estudo` já tiver arquivos, use outro nome em `--output` e nos comandos abaixo.

## 5. Gerar texto

```powershell
.\.venv\Scripts\python.exe generate.py --checkpoint runs/estudo/last.pt --prompt "Hello" --max-new-tokens 30 --device cpu
```

## 6. Avaliar

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/estudo/last.pt --data data/estudo --batches 2 --device cpu
```

A avaliação mostra a perda de validação (`val_loss`) e a perplexidade.

## Arquivos e outras opções

- `train_gpt2.py`: modelo, atenção causal, carregamento dos dados e treinamento.
- `prepare_data.py`: preparação de um texto local.
- `generate.py` e `evaluate.py`: geração e avaliação dos checkpoints.
- `fineweb.py`: preparação do FineWeb-Edu por streaming.
- `tests/`: testes da arquitetura, dos dados e do treino.

Use `--help` em qualquer script para consultar os argumentos. Algumas opções de treino:

- `--attention manual`: explicita os cálculos da atenção; o padrão usa SDPA do PyTorch.
- `--preset gpt2`: usa o GPT-2 completo, com 124.439.808 parâmetros; exige mais memória e tempo.
- `--resume caminho/last.pt`: retoma o treino, mantendo os mesmos dados e parâmetros do comando original.

O [vídeo da atividade](https://youtu.be/l8pRSuU81PU) é a referência indicada.
A conferência técnica registrada em `VALIDACAO.md` usou o código e o histórico do GitHub;
o conteúdo audiovisual não foi acessado. O relatório e a reflexão individual são entregas separadas,
conforme [requisitos.md](requisitos.md).
