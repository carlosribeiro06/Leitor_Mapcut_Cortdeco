# Leitor Mapcut / Cortdeco

Lê os binários **`mapcut`** e **`cortdeco`** de saída do DECOMP usando a biblioteca
[`idecomp`](https://github.com/rjmalves/idecomp) e exporta todo o conteúdo em CSV.

O `mapcut` é o cabeçalho da função de custo futuro (árvore de cenários, usinas,
estágios, tempo de viagem da água, dados de GNL, geometria dos registros de corte).
O `cortdeco` guarda os cortes de Benders propriamente ditos — um registro de tamanho
fixo por corte, com o termo constante (`rhs`) e os multiplicadores duais.

---

## Início rápido

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .

# lê mapcut.rv0 + cortdeco.rv0 e grava os CSVs em ./output
.venv/bin/python -m leitor_mapcut_cortdeco
```

Ou, com o pacote instalado, pelo comando de console:

```bash
leitor-mapcut-cortdeco --help
leitor-mapcut-cortdeco --settings /caminho/settings.json
leitor-mapcut-cortdeco --output-directory ./saida_rv0 --log-level DEBUG
leitor-mapcut-cortdeco --somente-mapcut
```

Nada é passado por linha de comando além de *qual* configuração usar, onde gravar e o
nível de log. Todo o resto (nomes dos arquivos, formato dos CSVs, geometria) vive no
`settings.json`, para que uma execução oficial seja reproduzível a partir de um arquivo
versionado, e não de uma linha de comando digitada na hora.

---

## Configuração (`settings.json`)

Os caminhos relativos são resolvidos **em relação ao diretório do próprio
`settings.json`**, e não ao diretório de trabalho de quem chamou o programa — assim o
projeto roda sem alteração em WSL, em servidor Linux ou na máquina de outra pessoa.
A validação é *fail fast*: qualquer chave ausente, com tipo errado ou fora do domínio
aceito interrompe a execução apontando exatamente a chave problemática.

| Seção | Chave | Significado |
| --- | --- | --- |
| `paths` | `input_directory` | Diretório dos binários de entrada. |
| | `mapcut_file`, `cortdeco_file` | Nomes dos binários (só o nome, sem diretório). |
| | `output_directory` | Destino dos CSVs. |
| `logging` | `level` | `DEBUG` … `CRITICAL`. |
| | `directory`, `filename` | Arquivo de log de auditoria. |
| | `max_bytes`, `backup_count` | Política de rotação do log. |
| | `use_rich` | Usa `RichHandler` no console quando o `rich` está instalado. |
| `export` | `separator`, `decimal` | Padrão `,` e `.` (portável para pandas/R). Para abrir no Excel pt-BR, use `;` e `,`. |
| | `float_format` | `null` preserva a precisão total do `float64`. |
| | `encoding` | Codificação dos CSVs. |
| | `write_mapcut` | Liga/desliga as tabelas do mapcut. |
| | `write_cortdeco_wide` | Liga/desliga o CSV *wide* dos cortes. |
| | `write_cortdeco_tidy` | Liga/desliga os CSVs longos por bloco de coeficientes. |
| `cortdeco` | `cuts_per_node_source` | `numero_iteracoes` (padrão) ou `file_geometry`. |
| | `cuts_per_node_override` | Inteiro para forçar o valor na mão, ou `null`. |

As chaves iniciadas por `_` no `settings.json` são comentários e são ignoradas na leitura.

---

## CSVs gerados

Vinte arquivos, com as contagens do caso `rv0` deste repositório (169 UHEs, 7 estágios,
273 nós, 73 iterações, 439 registros de corte):

### Do `mapcut`

| Arquivo | Linhas × Colunas | Conteúdo |
| --- | --- | --- |
| `mapcut_metadados.csv` | 11 × 3 | Escalares do caso, com descrição de cada um. |
| `mapcut_usinas_hidraulicas.csv` | 169 × 4 | `posicao`, `codigo_usina`, `indice_usina_jusante`, `codigo_usina_jusante`. A `posicao` é a mesma ordem dos coeficientes de volume armazenado dentro do corte. |
| `mapcut_arvore_cenarios.csv` | 273 × 3 | `no`, `estagio`, `no_pai`. |
| `mapcut_estagios.csv` | 7 × 3 | Primeiro nó e patamares de carga por estágio. |
| `mapcut_ultimo_corte_por_no.csv` | 273 × 3 | Índice do último corte de cada nó — o ponto de entrada da lista encadeada no `cortdeco`. |
| `mapcut_tempo_viagem_lags.csv` | 14 × 3 | Lag máximo por usina e estágio. |
| `mapcut_tempo_viagem.csv` | 56 × 5 | Coeficientes de amortecimento por usina, estágio e lag. **Corrigido** (ver defeito 2). |
| `mapcut_gnl.csv` | 14 × 5 | Dados de GNL por estágio e submercado. **Corrigido** (ver defeito 3). |
| `mapcut_gnl_bloco_valores.csv` | 98 × 3 | Bloco de reais do registro 9, indexado pela posição. |
| `mapcut_submercados_gnl.csv` | 2 × 1 | Submercados com despacho antecipado. |
| `mapcut_custos.csv` | 7 × 7 | Taxa de desconto e parcelas de custo por estágio. |

### Do `cortdeco`

| Arquivo | Linhas × Colunas | Conteúdo |
| --- | --- | --- |
| `cortdeco_cortes.csv` | 439 × 221 | **Wide**: uma linha por corte, uma coluna por coeficiente. A forma mais próxima do registro binário. |
| `cortdeco_rhs.csv` | 439 × 4 | Só o termo constante de cada corte. |
| `cortdeco_coef_volume_armazenado.csv` | 74 191 × 5 | Tidy: `codigo_usina`, `valor`. |
| `cortdeco_coef_defluencia_tempo_viagem.csv` | 2 634 × 6 | Tidy: `codigo_usina`, `lag`, `valor`. |
| `cortdeco_coef_geracao_gnl.csv` | 18 438 × 7 | Tidy: `codigo_submercado`, `estagio_coeficiente`, `patamar`, `valor`. **Corrigido** (ver defeito 4). |

### Cópias brutas para auditoria

Três CSVs com sufixo `_idecomp_bruto` reproduzem a saída **literal** da biblioteca, para
que dê para conferir exatamente o que foi corrigido:
`mapcut_usinas_jusante_idecomp_bruto.csv` (169 × 3),
`mapcut_tempo_viagem_idecomp_bruto.csv` (84 × 5),
`mapcut_gnl_idecomp_bruto.csv` (2 × 5) e
`cortdeco_coef_geracao_gnl_idecomp_bruto.csv` (18 438 × 6).

### Colunas do formato wide

```
indice_corte, no, estagio, rhs,
pi_varm_uhe{codigo},                       # um por UHE
pi_qdefp_uhe{codigo}_lag{0..L-1},          # UHE com tempo de viagem × lag
pi_gnl_sbm{submercado}_pat{patamar}_lag{estagio}
```

> Atenção ao último bloco: no nome gerado pelo `idecomp`, o token `lag` enumera
> **estágios** (1 a `numero_estagios`) e o token `pat` enumera **patamares de carga**.
> O CSV tidy separa as duas dimensões em colunas próprias.

---

## Lendo os CSVs de volta

A gravação preserva o valor exato: com `float_format: null`, o pandas grava a menor
representação decimal que reconstrói o `float64` original. Mas o **leitor** padrão do
pandas usa um parser rápido que perde os últimos bits. Como os coeficientes são duais de
um problema de otimização, quem for reconstruir a função de custo futuro a partir do CSV
deve pedir a leitura exata:

```python
import pandas as pd

cortes = pd.read_csv("output/cortdeco_cortes.csv", float_precision="round_trip")
```

Com `float_precision="round_trip"` a ida e volta é idêntica bit a bit (verificado em
teste). Sem ele, o erro relativo fica na ordem de 1e-15 — irrelevante para inspeção,
relevante para reconstrução numérica.

---

## Como a leitura do `cortdeco` é conferida

O `cortdeco` não é autodescritivo: os cortes de um mesmo nó formam uma **lista
encadeada** (o primeiro campo `int32` de cada registro aponta para o próximo, e 0 encerra
a cadeia), e toda a geometria vem do `mapcut`. Se a geometria estiver errada, a leitura
**não falha** — ela devolve números plausíveis e errados. Por isso o leitor confere três
coisas antes de ler qualquer corte:

1. **Tamanho múltiplo do registro.** O arquivo tem de ser um número inteiro de registros.
   Caso contrário, o par mapcut/cortdeco provavelmente não é do mesmo caso. → erro.
2. **Contagem de registros.** Tem de valer
   `cortes_por_nó × nós_que_constroem_corte + 1`. O `+ 1` é um registro extra que o
   DECOMP grava para o último estágio que constrói cortes. → aviso.
3. **Preenchimento nulo.** O registro é superdimensionado pelo DECOMP: só
   `4 + 8 × número_de_coeficientes` bytes são usados e o resto é zero. Se algum byte do
   preenchimento não for nulo, a contagem de coeficientes está subestimada e a leitura
   estaria truncando dados reais. → erro.

No caso `rv0` deste repositório: registro de 26 976 bytes, 218 coeficientes úteis
(1 748 bytes) e 25 228 bytes de preenchimento — verificados nulos em todos os 439
registros.

### Número de cortes por nó

Esse valor dimensiona os vetores de leitura e **não está gravado explicitamente** no
mapcut: ele equivale ao número de iterações da política. Um erro aqui é especialmente
traiçoeiro — se for superestimado, sobram linhas inteiramente zeradas; se subestimado,
faltam cortes. Por isso ele é obtido por duas vias independentes, sempre comparadas
entre si e registradas no log:

- `mapcut.numero_iteracoes`;
- geometria do arquivo: `(número_de_registros − 1) / nós_que_constroem_corte`.

No caso `rv0`: `numero_iteracoes = 73` e `(439 − 1) / 6 = 73` — as duas vias concordam.
Note que **não** é `mapcut.numero_cortes` (438), que é o total somando todos os nós;
usá-lo produziria 438 linhas nulas de preenchimento por nó.

---

## Defeitos corrigidos do `idecomp` 1.14.2

Quatro propriedades da biblioteca decodificam esses binários de forma incorreta. Em todos
os casos o CSV principal traz o dado corrigido, uma cópia `_idecomp_bruto` preserva a
saída literal da biblioteca, e a correção é registrada no log.

**1. `Mapcut.codigos_uhes_jusante` — tipo errado.**
O registro 4 guarda a topologia da cascata como `int32`, mas a biblioteca o lê como
`float32`; o resultado é uma lista de denormais sem sentido (`3e-45`, `4e-45`, …).
Reinterpretando os bits de volta como `int32`, os valores ficam todos em
`[0, número_de_UHEs]`. O valor é um **índice posicional 1-based** na lista `codigos_uhes`
(não um código de usina — há valores que não são códigos válidos), e 0 significa que a
usina não tem jusante no caso. Conferido contra topologia conhecida do SIN:
Camargos (1) → Itutinga (2) → Funil-Grande (4).

**2. `Mapcut.dados_tempo_viagem` — acumuladores não reiniciados.**
As listas de lag e de coeficiente não são zeradas a cada usina, e o offset do bloco
seguinte é recalculado em vez de acumulado. No caso `rv0` isso devolve 84 linhas em vez
de 56 (2 UHEs × 7 estágios × 4 lags): o bloco da segunda usina vem prefixado com os 28
valores da primeira, e os rótulos de estágio saem deslocados (8 por estágio em vez de 4).
A correção percorre o payload bruto com offset cumulativo. Como efeito colateral, a
propriedade original só é correta para casos com no máximo duas usinas com tempo de
viagem.

**3. `Mapcut.dados_gnl` — passo de registro errado.**
O passo entre os registros de cada estágio é calculado como `1 + 4n + Σpatamares`, mas o
registro realmente ocupa `1 + 3n` inteiros mais `numero_estagios × n` reais — no caso
`rv0`, 15 em vez de 21. Com o passo errado, só o primeiro estágio é decodificado
(2 linhas em vez de 14). A correção usa o passo verdadeiro.

**4. `Cortdeco.coeficientes_geracao_gnl` — rótulo trocado.**
A coluna `lag` é extraída do token `pat` do nome da coluna, de modo que o rótulo sai
errado e a dimensão de estágio desaparece: os 42 coeficientes de cada corte colapsam em
três valores ambíguos. A tabela corrigida deriva as três dimensões diretamente dos nomes
das colunas do formato wide.

Os blocos de volume armazenado e de defluência por tempo de viagem são decodificados
corretamente pela biblioteca e usam as propriedades originais sem alteração.

---

## Log de auditoria

Cada execução escreve simultaneamente no console (via `rich`, quando disponível) e em um
arquivo rotativo. O arquivo é a trilha de auditoria: ele basta para reconstruir o que uma
execução oficial fez — binários de entrada e seus tamanhos, geometria derivada e como ela
foi conferida, número de cortes por nó e de onde veio, contagem de linhas de cada CSV, e
o tempo decorrido de cada etapa.

```
2026-09-04 11:52:01 | INFO | ... | geometria do corte: 26976 bytes por registro, 218 coeficientes uteis (1748 bytes) + 25228 bytes de preenchimento
2026-09-04 11:52:01 | INFO | ... | registros: 439 no arquivo; 6 nos constroem cortes x 73 cortes por no + 1 registro extra = 439 esperados (origem dos cortes por no: numero_iteracoes)
2026-09-04 11:52:01 | INFO | ... | integridade confirmada: os 25228 bytes de preenchimento de cada um dos 439 registros sao inteiramente nulos
```

Divergências não passam em branco: `WARNING` para contagem de registros inesperada,
desacordo entre as duas vias de cortes por nó, uso de `cuts_per_node_override`, cortes
inteiramente zerados e lags de tempo de viagem não uniformes.

---

## Desenvolvimento

```bash
uv pip install --python .venv/bin/python -e ".[dev]"

.venv/bin/python -m pytest -q      # 70 testes
.venv/bin/ruff check src tests
.venv/bin/ruff format src tests
.venv/bin/mypy src                 # strict
```

Os testes usam duas estratégias. O **caso sintético** é um `cortdeco` de 7 registros
gravado byte a byte, em que cada coeficiente carrega no próprio valor o registro e a
posição de onde veio (`registro × 100 + posição`) — isso permite afirmar que *cada*
coeficiente lido veio do registro certo, e não apenas que a tabela tem o tamanho
esperado. Os **testes de integração** rodam sobre os binários reais da raiz do
repositório, e são ignorados automaticamente quando eles não estão presentes.

---

## Limitações conhecidas

- **Patamares uniformes.** O leitor de cortes do `idecomp` aplica um único número de
  patamares de carga a todos os estágios. Se o caso tiver estágios com números
  diferentes, a largura do bloco de coeficientes GNL não pode ser determinada com
  segurança e a execução para com mensagem explícita, em vez de escolher um valor.
- **Semântica do bloco de reais do GNL.** O bloco final do registro 9 do mapcut é
  dimensionado pelo DECOMP como `numero_estagios × numero_utes_gnl`, mas em geral só as
  primeiras `numero_patamares × numero_utes_gnl` posições vêm preenchidas. O significado
  exato de cada posição não está documentado, então ele é exportado indexado pela posição
  dentro do bloco, sem rótulos inventados.
- **Versões fixadas.** As correções dos defeitos 2 e 3 leem o payload bruto da seção de
  dados do mapcut. Por isso `idecomp` e `cfinterface` estão fixados em versão exata no
  `pyproject.toml`; ao subir de versão, rode a suíte de testes antes de usar os CSVs em
  estudo oficial.
