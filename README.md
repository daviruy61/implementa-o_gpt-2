# Implementação do GPT-2

Projeto de estudo baseado em [build-nanogpt, de Andrej Karpathy](https://github.com/karpathy/build-nanogpt)(https://www.youtube.com/watch?v=l8pRSuU81PU).
A arquitetura e o treinamento estão em `train_gpt2.py`.

## 1. Instalar

Use Python 3.12 e abra o PowerShell na pasta do projeto:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se o ambiente já existe, basta o segundo comando. Não é necessário ativar a `.venv`.
Todas as dependências estão no mesmo arquivo, incluindo Transformers e datasets.

O PyTorch faz as operações com tensores, calcula gradientes e atualiza os pesos.
O NumPy armazena os tokens e o tiktoken fornece o tokenizador GPT-2.
Transformers permite importar pesos oficiais, datasets carrega o FineWeb-Edu e pytest executa os testes.
O exemplo abaixo treina a implementação local a partir de pesos aleatórios.

## 2. Executar os testes

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Os testes não baixam corpus nem pesos. Eles verificam dimensões dos tensores,
atenção causal, cálculo da perda, gradientes, geração e retomada do treinamento.
Por exemplo, alterar um token futuro não deve mudar os logits das posições anteriores:
isso confere se a máscara causal impede o modelo de consultar a resposta durante o treino.

## 3. Preparar o texto

```powershell
.\.venv\Scripts\python.exe prepare_data.py --input examples/corpus.txt --output data/estudo
```

O script usa o tokenizador GPT-2 e separa 90% dos tokens para treino e 10% para validação.
A primeira execução pode baixar o vocabulário. Para usar outro texto, troque o caminho de `--input`.
Se `data/estudo` já foi preparada nesta versão, pule esta etapa ou escolha outra pasta.

**O que acontece com o texto:** o BPE transforma o texto em IDs de um vocabulário
de 50.257 tokens. Um token pode representar uma palavra, parte dela ou pontuação;
portanto, contar tokens não é o mesmo que contar palavras. O tokenizador já está
definido e não é aprendido neste treinamento.

O script guarda os IDs em arquivos `.npy` como inteiros `uint16`, suficientes para
esse vocabulário. O trecho final fica reservado para validação. O `DataLoaderLite`
converte os IDs para `torch.long` e monta entradas e alvos deslocados em uma posição:

```text
Sequência de IDs: [10, 20, 30, 40]
Entrada x:       [10, 20, 30]
Alvo y:          [20, 30, 40]
```

Os números são ilustrativos. Na primeira posição, o modelo vê `10` e tenta prever
`20`; na segunda, vê o contexto `[10, 20]` e tenta prever `30`.
Assim, cada posição da entrada fornece um exemplo de previsão do próximo token.

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

### Como os dados atravessam o modelo

Nos formatos abaixo, `B` é o tamanho do lote, `T` o número de tokens da sequência,
`C` a largura dos vetores e `V` o tamanho do vocabulário. No comando acima,
`B=1`, `T=8`, `C=64` e `V=50257`. O `tiny` tem 2 blocos, 4 cabeças de atenção e
contexto máximo de 64 tokens; os 8 tokens usados no treino cabem nessa janela.

```text
IDs [B,T] -> embeddings [B,T,C] -> blocos Transformer [B,T,C]
          -> LayerNorm final -> projeção para logits [B,T,V]
```

| Componente no código | Operação e finalidade |
|---|---|
| `wte` e `wpe` | Somam um vetor aprendido para cada token e outro para sua posição. O primeiro representa o token; o segundo informa sua ordem na sequência. A saída é `[B,T,C]`. |
| `CausalSelfAttention` | Combina informações do token atual e dos anteriores, dando pesos diferentes às posições conforme o contexto. Mantém o formato `[B,T,C]`. |
| `LayerNorm` | Calcula a média e a variância dos `C` canais de cada token para normalizá-los, depois aplica escala e deslocamento aprendidos. Ajuda a controlar a escala das ativações e aparece antes de cada ramo do bloco. |
| `MLP` | Aplica `C → 4C → C`, com GELU entre as duas projeções. Essa não linearidade transforma as características de cada posição, sem misturar tokens diretamente. |
| `Block` e conexões residuais | Fazem `x = x + atenção(LayerNorm(x))` e depois `x = x + MLP(LayerNorm(x))`. As somas preservam a informação anterior e oferecem um caminho direto para os gradientes. |
| `ln_f` e `lm_head` | Normalizam a saída dos blocos e a projetam para `V` escores por posição. A projeção compartilha a matriz de pesos de `wte` (*weight tying*), reduzindo o número de parâmetros. |

**Dentro da atenção:** uma projeção linear produz consultas (`Q`), chaves (`K`) e
valores (`V`). Aqui, `V` designa os valores da atenção, não o tamanho do vocabulário.
As consultas e chaves determinam a relevância entre posições; os valores carregam
a informação que será combinada. Cada tensor é dividido em `H` cabeças, com
formato `[B,H,T,D]`, onde `D=C/H`. No exemplo, são `[1,4,8,16]`.

```text
atenção(Q, K, V) = softmax(QKᵀ / sqrt(D) + máscara) V
```

`QKᵀ` produz escores `[B,H,T,T]`. A divisão por `sqrt(D)` controla sua escala,
evitando distribuições excessivamente concentradas só pelo aumento da dimensão.
A máscara vale zero no passado e na posição atual, e menos infinito no futuro;
após o softmax, posições futuras têm peso zero. O softmax é aplicado sobre as
posições consultadas, e a multiplicação por `V` faz a soma ponderada dos valores.
As cabeças permitem combinações diferentes do contexto; suas saídas são concatenadas
e projetadas de volta para `C` canais.

O padrão usa `scaled_dot_product_attention` (SDPA) do PyTorch. Acrescentar
`--attention manual` executa esses cálculos explicitamente. As duas opções realizam
a mesma operação; o uso de um kernel como FlashAttention depende do ambiente.

### Como o modelo aprende

Para uma entrada de formato `[1,8]`, a saída tem formato `[1,8,50257]`: em cada uma das oito
posições há um escore, chamado **logit**, para cada token possível. Logits ainda
não são probabilidades.

1. A **entropia cruzada** compara os logits com os alvos deslocados. Ela calcula
   a média de `-log(p do token correto)` nas posições do lote: atribuir pouca
   probabilidade ao alvo aumenta a perda. `cross_entropy` já inclui o log-softmax;
   não se aplica softmax antes de passar os logits para essa função.
2. `loss.backward()` calcula os gradientes pela regra da cadeia, indicando como
   cada peso influencia a perda. Os gradientes são zerados no início de cada etapa.
3. O **clipping** limita a norma global para controlar gradientes muito grandes.
   O **AdamW** usa esses gradientes para atualizar os pesos; o decaimento
   de pesos é aplicado às matrizes, mas não aos vieses nem aos parâmetros da LayerNorm.
4. A taxa de aprendizado sobe linearmente nas duas etapas de **warmup** e depois
   diminui seguindo uma curva de cosseno. Isso controla o tamanho das atualizações.

`--max-steps 10` significa dez atualizações, não dez passagens completas pelo corpus.
Nesse comando, cada atualização usa `1 × 8 = 8` tokens. Com
`--grad-accum-steps N`, o treino acumula gradientes de `N` lotes menores, dividindo
a perda por `N`, antes de atualizar os pesos; isso permite aumentar o lote efetivo.

O checkpoint `last.pt` guarda pesos, configuração, estado do otimizador, etapa e
posição do carregador. Essas informações permitem continuar um treino interrompido.

## 5. Gerar texto

```powershell
.\.venv\Scripts\python.exe generate.py --checkpoint runs/estudo/last.pt --prompt "Hello" --max-new-tokens 30 --device cpu
```

O prompt é tokenizado e passa pelo modelo sem calcular gradientes nem atualizar pesos.
A geração usa apenas os logits da **última posição**, pois eles predizem o token
que vem depois do contexto. O token sorteado é acrescentado à sequência, e o
processo se repete 30 vezes: isso é geração **autoregressiva**.

Por padrão, a temperatura é `1.0` e o `top-k` é `50`. O código calcula
`softmax(logits / temperatura)`, mantém os 50 tokens mais prováveis e sorteia um
deles conforme seus pesos. Temperaturas positivas menores que 1 concentram a
distribuição; maiores que 1 a tornam mais uniforme. Com `--temperature 0`, o
programa escolhe diretamente o maior logit. Se o contexto ultrapassar o limite
do modelo, apenas os tokens mais recentes entram na próxima previsão.

## 6. Avaliar

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint runs/estudo/last.pt --data data/estudo --batches 2 --device cpu
```

A avaliação usa os tokens reservados para validação, sem calcular gradientes nem
alterar os pesos. `--batches 2` mede até dois lotes completos, não necessariamente
todo o conjunto de validação.

- **`val_loss`**: média da entropia cruzada nos lotes avaliados. Mede a penalidade
  por atribuir pouca probabilidade aos próximos tokens observados nesse trecho.
- **Perplexidade**: `exp(val_loss)`. Pode ser interpretada como um número efetivo
  de alternativas por previsão. Valores menores indicam previsões melhores quando
  se compara o mesmo conjunto de dados e tokenizador; não é uma porcentagem de acerto.

Comparar a perda de treino com a de validação ajuda a perceber se o modelo melhora
também em texto que não usou para atualizar os pesos. Os dez passos deste exemplo
verificam a execução, mas não demonstram aprendizado suficiente para gerar texto coerente.

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
