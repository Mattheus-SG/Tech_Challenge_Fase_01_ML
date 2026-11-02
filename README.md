# Predição de Churn de Clientes (Telco): Modelagem Clássica e API REST

Projeto do **Tech Challenge da Fase 1 (12MLET, POSTECH)**, desenvolvido pelo **Grupo 28**.

Um modelo de Machine Learning que classifica clientes de uma operadora de telecomunicações pelo risco de cancelamento (*churn*), servido por uma API REST. O trabalho vai da análise exploratória até o modelo em produção local, comparando um modelo linear, um baseado em árvores e uma rede neural simples, todos com scikit-learn.

## 1. Sobre o projeto

**Problema de negócio.** Uma operadora está perdendo clientes em ritmo acelerado, e a diretoria precisa de um modelo que aponte quais clientes têm maior risco de cancelar, para que a equipe de retenção priorize suas ações.

**Dataset.** IBM Telco Customer Churn (dataset público da IBM), com 7.043 clientes e 33 colunas. 26,5% dos clientes cancelaram. O CSV já está em `data/`, não é preciso baixá-lo.

**Métricas.** F1 da classe churn (principal), AUC-ROC (secundária) e precision e recall (apoio). Como métrica de negócio, a parcela da cobrança mensal dos clientes que cancelaram que o modelo consegue sinalizar.

**Decisões técnicas principais:**

- **Limpeza:** remoção das colunas com risco de vazamento do alvo (`Churn Label`, `Churn Score`, `Churn Reason` e, por precaução, `CLTV`), das constantes e das geográficas. O `TotalCharges` em branco (11 clientes novos, com `tenure` 0) foi preenchido com 0.
- **Pré-processamento dentro do `Pipeline`:** `StandardScaler` nas numéricas e `OneHotEncoder` nas categóricas, de modo que o treino e a API usam exatamente a mesma transformação.
- **Modelo final:** Random Forest (200 árvores, profundidade máxima 10, `class_weight="balanced"`), escolhido por `GridSearchCV` com validação cruzada de 5 folds.
- **Limiar de decisão:** 0,5, definido em `src/config.py`.
- **API:** o modelo é carregado na subida, a entrada é validada com Pydantic e a latência de cada requisição vai no cabeçalho `X-Process-Time`.

## 2. Resultados

| Modelo | F1 (CV) | F1 (teste) | AUC (teste) | Precision | Recall |
|---|---|---|---|---|---|
| **Random Forest (campeão)** | **0,649 ± 0,016** | **0,637** | 0,852 | 0,540 | **0,778** |
| Rede Neural (MLP) | 0,621 ± 0,031 | 0,614 | 0,853 | 0,636 | 0,594 |
| Regressão Logística balanceada | 0,640 ± 0,022 | 0,618 | 0,849 | 0,512 | 0,778 |
| Regressão Logística (baseline) | 0,620 ± 0,031 | 0,606 | 0,849 | 0,645 | 0,572 |

Métricas de teste em 1.409 clientes (divisão 80/20 estratificada, seed 42). O campeão identifica 78% dos clientes que cancelaram (291 de 374), e 54% dos clientes que ele sinaliza realmente cancelariam.

Os modelos têm AUC equivalente. Boa parte da vantagem do Random Forest sobre o baseline vem do `class_weight="balanced"`: com os mesmos pesos, a Regressão Logística chega perto (+0,009 de F1 na validação cruzada e +0,020 no teste, com o mesmo recall e 29 alarmes falsos a menos para o Random Forest). A escolha do campeão se mantém, mas a vantagem é pequena.

O baseline do notebook 01 tem F1 0,642 no teste, em outra divisão (seed 123), com 17 features e `class_weight="balanced"`. Esse valor não é comparável com os da tabela acima, em que todos os modelos usam a mesma divisão (seed 42).

A análise completa, incluindo as limitações e o desempenho por grupo, está no [Model Card](docs/MODEL_CARD.md).

## 3. Etapas do trabalho e onde estão

| Etapa | Entregável | Onde |
|---|---|---|
| 1. Entendimento e preparação | ML Canvas, EDA, métricas e baseline de Regressão Logística | `notebooks/01_eda_e_baseline.ipynb` |
| 2. Modelagem e avaliação | Random Forest e MLP, validação cruzada, tabela comparativa e modelo campeão salvo | `notebooks/02_modelagem_e_avaliacao.ipynb` e `models/` |
| 3. Engenharia de software e API | Código refatorado em módulos, testes com pytest e API FastAPI | `src/` e `tests/` |
| 4. Documentação e entrega final | Model Card e README | `docs/MODEL_CARD.md` e `README.md` |
| 4. Documentação e entrega final | Vídeo de apresentação (método STAR) | Entregue à parte, fora do repositório |

## 4. Estrutura do repositório

```
.
├── data/
│   └── IBM_Telco_customer_churn_IBM_dataset.csv   # dataset original
├── docs/
│   └── MODEL_CARD.md                              # Model Card do modelo campeão
├── models/
│   ├── 02_modelo_campeao.joblib                   # modelo servido pela API (Pipeline completo)
│   ├── 02_modelo_campeao_metadata.json            # metadados do campeão
│   ├── 02_resultado_treinamento.csv               # tabela comparativa dos modelos
│   ├── 02_experimentos_gridsearch.csv             # registro das 76 combinações testadas
│   ├── 02_log_experimentos.txt                    # log dos experimentos
│   ├── 02_avaliacao_campeao.json                  # métrica de negócio, subgrupos, treino x teste e baseline balanceado
│   └── baseline_*                                 # baseline da Etapa 1 (modelo, métricas, configuração)
├── notebooks/                                     # experimentação (Etapas 1 e 2)
├── outputs/                                       # gráficos gerados pelo notebook 01
├── src/
│   ├── api/
│   │   ├── main.py                                # app FastAPI (/health e /predict)
│   │   └── schemas.py                             # validação de entrada e saída (Pydantic)
│   ├── config.py                                  # seeds, caminhos, features e hiperparâmetros
│   ├── preprocessing.py                           # carga e limpeza dos dados
│   ├── pipeline.py                                # Pipeline do modelo campeão
│   ├── train.py                                   # treino de ponta a ponta (linha de comando)
│   ├── evaluate.py                                # números do Model Card (linha de comando)
│   └── predict.py                                 # inferência
├── tests/                                         # testes automatizados (pytest)
├── Dockerfile                                     # imagem da API para rodar em container
├── .dockerignore                                  # limita a imagem ao que a API precisa
├── pyproject.toml                                 # dependências e configuração de pytest e ruff
├── uv.lock                                        # versões exatas das dependências
└── .python-version                                # versão do Python (3.14)
```

A experimentação fica nos notebooks, e o código de produção fica em módulos `.py` dentro de `src/`.

## 5. Instalação

**Pré-requisitos:** Git e o gerenciador [uv](https://docs.astral.sh/uv/). O projeto exige Python 3.14, e o `uv` o instala sozinho se você não tiver.

**1. Instale o uv** (se ainda não tiver):

```bash
# macOS e Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Depois de instalar, feche e abra o terminal. Outras formas de instalação estão na [documentação do uv](https://docs.astral.sh/uv/getting-started/installation/).

**2. Clone o repositório:**

```bash
git clone https://github.com/Mattheus-SG/Tech_Challenge_Fase_01_ML.git
cd Tech_Challenge_Fase_01_ML
```

**3. Instale as dependências:**

```bash
uv sync
```

O comando cria o ambiente virtual em `.venv/` e instala as versões exatas registradas em `uv.lock`, incluindo as ferramentas de desenvolvimento (pytest, ruff, jupyter e outras). Para instalar só o necessário para rodar a API, use `uv sync --no-dev`.

**4. Confira a instalação (opcional):**

```bash
uv run pytest -m "not slow"
```

## 6. Como rodar a API localmente

Na raiz do repositório:

```bash
uv run uvicorn src.api.main:app --reload
```

A API sobe em `http://127.0.0.1:8000`, e a documentação interativa (Swagger) fica em **http://127.0.0.1:8000/docs**. Acessar a raiz (`/`) redireciona para ela. Para parar, use `Ctrl+C`. Para usar outra porta, acrescente `--port 8001`.

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/health` | Informa se a API está no ar e se o modelo está carregado |
| `POST` | `/predict` | Recebe os dados de um cliente e devolve a propensão de churn |

### Exemplo: `/health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{
  "status": "healthy",
  "modelo_carregado": true,
  "modelo": "Random Forest (Tuning + KBest)",
  "versao_modelo": "2026-10-25T19:49:55",
  "versao_api": "0.1.0"
}
```

Se o modelo não puder ser carregado, o `/health` responde `503` com `"status": "degraded"`.

### Exemplo: `/predict`

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 151.65
  }'
```

```json
{
  "propensao_churn": 0.8467,
  "churn": true,
  "limiar": 0.5,
  "versao_modelo": "2026-10-25T19:49:55"
}
```

O valor de `propensao_churn` está arredondado aqui, e a API devolve todas as casas decimais. O campo `churn` é `true` quando a propensão atinge o limiar de 0,5.

No Windows, o jeito mais simples de testar é o Swagger: abra `/docs`, clique em `POST /predict`, depois em **Try it out** e **Execute**. O corpo de exemplo já vem preenchido com o cliente acima.

### Campos de entrada

Todos os 19 campos são obrigatórios. Os valores categóricos são em inglês, exatamente como no dataset.

| Campo | Valores aceitos |
|---|---|
| `gender` | `Female`, `Male` |
| `SeniorCitizen` | `0` ou `1` (1 = 65 anos ou mais) |
| `Partner`, `Dependents` | `Yes`, `No` |
| `tenure` | inteiro maior ou igual a 0 (meses como cliente) |
| `PhoneService`, `PaperlessBilling` | `Yes`, `No` |
| `MultipleLines` | `Yes`, `No`, `No phone service` |
| `InternetService` | `DSL`, `Fiber optic`, `No` |
| `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | `Yes`, `No`, `No internet service` |
| `Contract` | `Month-to-month`, `One year`, `Two year` |
| `PaymentMethod` | `Bank transfer (automatic)`, `Credit card (automatic)`, `Electronic check`, `Mailed check` |
| `MonthlyCharges`, `TotalCharges` | número maior ou igual a 0 (US$) |

Duas regras de consistência são validadas: `MultipleLines` deve ser `No phone service` se, e somente se, `PhoneService` for `No`; e os seis serviços de internet devem ser `No internet service` se, e somente se, `InternetService` for `No`.

### Respostas de erro

| Código | Quando acontece |
|---|---|
| `422` | Campo ausente, valor fora dos aceitos (inclusive infinito) ou combinação de serviços impossível |
| `503` | O modelo não pôde ser carregado |
| `500` | Erro inesperado na inferência (a mensagem é genérica, e os detalhes ficam só no log do servidor) |

### Usando outro modelo

Por padrão a API serve `models/02_modelo_campeao.joblib`. Para servir outro arquivo (por exemplo, um modelo retreinado), defina a variável `MODEL_PATH` a partir da raiz do repositório. O arquivo de metadados `<nome>_metadata.json` precisa estar na mesma pasta.

```bash
# macOS e Linux
MODEL_PATH=models/modelo_retreinado.joblib uv run uvicorn src.api.main:app --reload
```

```powershell
# Windows (PowerShell)
$env:MODEL_PATH = "models/modelo_retreinado.joblib"; uv run uvicorn src.api.main:app --reload
```

### Rodando com Docker

A imagem contém só a API, o modelo campeão e as dependências de produção, nas versões do `uv.lock`, e roda sem privilégios de root.

```bash
docker build -t churn-api .
docker run --rm -p 8000:8000 churn-api
```

A documentação interativa fica em **http://127.0.0.1:8000/docs**, como na execução local.

### Implantando na nuvem (opcional)

A API está publicada no Google Cloud Run: **https://churn-api-976469610767.southamerica-east1.run.app/docs**. O serviço pode ser desligado depois da avaliação do Tech Challenge.

A imagem funciona em serviços de containers como Google Cloud Run, Amazon ECS e Azure Container Apps. Ela escuta na porta da variável `PORT`, quando definida, ou na 8000. Com 512 MB de memória, ela atendeu 200 chamadas seguidas sem erros e usou cerca de 170 MB.

Para repetir a implantação no Google Cloud Run, é preciso ter Docker, um projeto do Google Cloud com cobrança ativa e a CLI `gcloud` autenticada, com esse projeto selecionado. Troque `SEU_PROJETO` pelo ID do projeto. O primeiro comando só é necessário na primeira implantação:

```bash
gcloud artifacts repositories create cloud-run-source-deploy --repository-format=docker --location=southamerica-east1
gcloud auth configure-docker southamerica-east1-docker.pkg.dev
docker build --platform linux/amd64 -t southamerica-east1-docker.pkg.dev/SEU_PROJETO/cloud-run-source-deploy/churn-api .
docker push southamerica-east1-docker.pkg.dev/SEU_PROJETO/cloud-run-source-deploy/churn-api
gcloud run deploy churn-api --image southamerica-east1-docker.pkg.dev/SEU_PROJETO/cloud-run-source-deploy/churn-api --region southamerica-east1 --memory 512Mi --max-instances 1 --allow-unauthenticated
```

Ao terminar, o último comando mostra o endereço público, e o Swagger fica em `<endereço>/docs`. A API não tem autenticação: com `--allow-unauthenticated`, qualquer pessoa com o endereço pode chamá-la, então use essa opção apenas em demonstrações, com dados fictícios. O `--max-instances 1` limita o custo se o serviço receber muitas chamadas.

## 7. Testes e qualidade de código

```bash
uv run pytest                      # todos os testes
uv run pytest -m "not slow"        # pula os testes que retreinam o modelo completo
uv run pytest --cov=src            # com cobertura de código
uv run ruff check .                # lint (configurado no pyproject.toml)
uv run ruff format --check .        # formatação
```

Os testes cobrem o pré-processamento, o pipeline, a inferência, a avaliação complementar e a API (`/health`, `/predict` e as respostas `422`, `503` e `500`). Os testes marcados como `slow` retreinam o modelo e conferem se ele reproduz as métricas da Etapa 2.

## 8. Notebooks

```bash
uv run jupyter lab
```

Abra os notebooks pela pasta `notebooks/`. O notebook 02 usa caminhos relativos (`../data`, `../models`) e depende de ser aberto a partir dela.

> **Cuidado:** executar os notebooks por inteiro regrava arquivos. O notebook 01 sobrescreve `outputs/*.png` e os arquivos `models/baseline_*`, e o notebook 02 sobrescreve os arquivos `models/02_*`, inclusive o modelo servido pela API. Como um Random Forest retreinado em outro ambiente é equivalente, mas não idêntico, as métricas podem mudar levemente. O notebook 02 também atualiza `models/02_avaliacao_campeao.json`. Se o modelo for trocado de outra forma, rode `uv run python -m src.evaluate`: até lá, o teste que confere esse arquivo é pulado, com uma mensagem explicando o motivo. Se quiser apenas retreinar, prefira o comando da próxima seção.

## 9. Retreinar e reavaliar o modelo

```bash
uv run python -m src.train --saida models/modelo_retreinado.joblib
```

O treino usa o mesmo pipeline, os mesmos hiperparâmetros e as mesmas seeds do modelo oficial, e salva o modelo e os metadados no caminho informado. O `--saida` é obrigatório para não sobrescrever o modelo oficial por engano. Os arquivos `models/modelo_retreinado*` são ignorados pelo Git. Para servi-lo, veja "Usando outro modelo".

Para regenerar a métrica de negócio, a tabela por subgrupo, a diferença entre treino e teste e a comparação com o baseline balanceado (`models/02_avaliacao_campeao.json`), sem alterar o modelo oficial:

```bash
uv run python -m src.evaluate
```

## 10. Reprodutibilidade

- **Seeds fixadas:** `SEED = 42` em `src/config.py`, usada na divisão treino/teste, na validação cruzada e nos modelos. O notebook 01 usa a seed 123, definida nele mesmo, e seu resultado não é comparável com os da seção 2.
- **Divisão e validação:** treino/teste de 80/20 estratificado e validação cruzada estratificada com 5 folds.
- **Dependências travadas:** `uv.lock` registra as versões exatas, e o `pyproject.toml` fixa `numpy`, `pandas` e `scikit-learn`.
- **Versão do scikit-learn:** o modelo campeão foi gerado com a 1.9.0 e deve ser carregado com a mesma versão. O notebook 01 foi executado com a 1.9.1, e `models/baseline_logistica.joblib` não é usado pela API. Como o formato é baseado em pickle, só carregue arquivos de origem confiável.

## 11. Model Card e limitações

O [Model Card](docs/MODEL_CARD.md) documenta a avaliação técnica detalhada, métricas de desempenho, análise de vieses por subgrupo e restrições operacionais. Os pontos de atenção e diretrizes de uso incluem:

- **Propensão:** O modelo fornece estimativas de propensão ao cancelamento para ordenação e priorização de contato, não garantindo que o cliente efetivamente cancelará o serviço.
- **Uso ético e humano:** Não deve ser utilizado como fundamento exclusivo para decisões punitivas, cancelamento preventivo ou aplicação automática de medidas prejudiciais aos clientes sem supervisão humana.
- **Qualidade e representatividade:** O desempenho depende da manutenção da qualidade e do formato dos dados de entrada, bem como do monitoramento contínuo de distribuições.
- **Vantagem do campeão:** é pequena em relação à Regressão Logística balanceada, e todos os modelos têm AUC equivalente.
- **Sobreajuste:** o modelo vai bem melhor no treino (F1 0,772) do que no teste (F1 0,637). As métricas de referência são as da validação cruzada e do teste, que concordam entre si.
- **Falsos alarmes e subgrupos:** Quase metade dos clientes sinalizados não cancelaria (46% de alarmes falsos no limiar de 0,5). Entre os clientes que permaneceram, 46% dos com 65 anos ou mais são sinalizados indevidamente, contra 21% dos demais, o que concentra as abordagens desnecessárias nesse grupo.

## 12. Equipe

Grupo 28:

- Derek Seiji Lima Sadakane
- Gustavo Eidi Kono Masuno
- Jônatas Gomes Alves
- Leandro Kersting de Freitas
- Matheus de Souza Gonçalves
