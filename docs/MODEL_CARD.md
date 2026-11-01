# Model Card —  Predição de Churn de Clientes (Telco)

## 1. Detalhes do modelo

| Campo | Valor |
|---|---|
| **Modelo** | Random Forest Classifier (Scikit-Learn / Tuning + KBest), campeão da Etapa 2 |
| **Tarefa** | Classificação binária: churn (1) ou permanência (0) |
| **Arquivos** | `models/02_modelo_campeao.joblib` e `models/02_modelo_campeao_metadata.json` |
| **Reprodutibilidade** | Seed 42, divisão treino/teste 80/20 estratificada, validação cruzada estratificada com 5 folds |

**Arquitetura.** Um único `Pipeline` do scikit-learn, para que o treino e a API usem exatamente o mesmo pré-processamento:

1. `ColumnTransformer`: `StandardScaler` nas 4 variáveis numéricas e `OneHotEncoder(drop="first", handle_unknown="ignore")` nas 15 categóricas.
2. `SelectKBest(f_classif, k="all")`: o GridSearch testou k = 10, 20 e `all`, e `all` venceu. Na prática, os 30 atributos gerados são usados.
3. `RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=2, class_weight="balanced")`.

**Como foi escolhido.** O GridSearchCV avaliou 76 combinações de hiperparâmetros (Random Forest e MLP), todas registradas em `models/02_experimentos_gridsearch.csv`. O critério foi o maior F1 médio na validação cruzada, com AUC-ROC como desempate. O conjunto de teste foi usado apenas para confirmar o resultado, não para decidir.

## 2. Uso pretendido

**Uso principal:** ordenar clientes ativos pela propensão de cancelamento, para que a equipe de retenção priorize quem contatar ou receber uma oferta. O público é a diretoria, a equipe de retenção e a equipe de dados.

**Fora do escopo:**

- Decisões automáticas, sem revisão humana, que limitem preço, acesso a serviço ou atendimento de um cliente.
- Interpretar a propensão como probabilidade de cancelamento (ela não é calibrada, ver seção 6).
- Explicar *por que* um cliente cancela: a importância das variáveis mostra associação, não causa.
- Usar o modelo em outra população (outra operadora, outro país ou outro período) sem revalidar.

## 3. Dados

| Item | Descrição |
|---|---|
| **Fonte** | IBM Telco Customer Churn (`data/IBM_Telco_customer_churn_IBM_dataset.csv`) |
| **Volume** | 7.043 clientes e 33 colunas |
| **Origem geográfica** | Todos os clientes são dos EUA (Califórnia): `Country` e `State` têm um único valor |
| **Alvo** | `Churn Value`: 1.869 clientes cancelaram (26,5%) |
| **Divisão** | 5.634 para treino e 1.409 para teste (374 churners e 1.035 que permaneceram) |
| **Variáveis usadas (19)** | Perfil (`gender`, `SeniorCitizen`, `Partner`, `Dependents`), tempo e cobrança (`tenure`, `MonthlyCharges`, `TotalCharges`), serviços contratados, tipo de contrato, fatura digital e forma de pagamento |

**Colunas descartadas:**

- **Vazamento de informação:** `Churn Label` repete o alvo, `Churn Reason` só existe para quem já cancelou e `Churn Score` é uma pontuação de churn já calculada. O `CLTV` foi descartado por precaução, porque não sabemos quando nem como foi calculado.
- **Constantes:** `Count`, `Country` e `State`.
- **Identificador e geografia:** `CustomerID`, `City`, `Zip Code`, `Lat Long`, `Latitude` e `Longitude`.

**Tratamentos:**

- `TotalCharges` vinha em branco em 11 clientes, todos com `tenure = 0`, e foi preenchido com 0. A premissa é verificada por código, mas não foi confirmada com a operadora.
- `SeniorCitizen` foi convertida de Yes/No para 1/0.

## 4. Desempenho

### 4.1 Comparação de modelos

Mesma divisão, mesma validação cruzada, limiar 0,5. A escolha usou as colunas de CV; as de teste só confirmam.

| Modelo | F1 (CV) | AUC (CV) | F1 (teste) | AUC (teste) | Precision | Recall | Acurácia |
|---|---|---|---|---|---|---|---|
| **Random Forest (campeão)** | **0,649 ± 0,016** | 0,859 ± 0,010 | **0,637** | 0,852 | 0,540 | **0,778** | 0,765 |
| Rede Neural (MLP) | 0,621 ± 0,031 | 0,859 ± 0,012 | 0,614 | 0,853 | 0,636 | 0,594 | 0,802 |
| Regressão Logística balanceada | 0,640 ± 0,022 | 0,859 ± 0,013 | 0,618 | 0,849 | 0,512 | 0,778 | 0,744 |
| Regressão Logística (baseline) | 0,620 ± 0,031 | 0,859 ± 0,013 | 0,606 | 0,849 | 0,645 | 0,572 | 0,803 |

Precision, recall e acurácia são da classe churn / do conjunto de teste.

**Como ler a tabela.** Todos os modelos têm AUC praticamente igual, ou seja, ordenam os clientes com qualidade equivalente. A vantagem do Random Forest está no perfil de erro: ele encontra mais cancelamentos (recall 0,78 contra 0,57 do baseline) e, em troca, erra mais ao sinalizar (precision 0,54 contra 0,64). Esse efeito vem em boa parte do `class_weight="balanced"`. O ganho de F1 na CV sobre o baseline (+0,029) é menor que dois desvios-padrão entre folds.

**Comparação com os mesmos pesos.** A linha "Regressão Logística balanceada" usa o mesmo `class_weight` do campeão, a mesma divisão e os mesmos folds. Com ela, a vantagem do Random Forest cai para +0,009 de F1 na CV (menos de um desvio-padrão) e +0,020 no teste, onde os dois têm o mesmo recall e o Random Forest gera 29 alarmes falsos a menos (248 contra 277). A escolha do campeão se mantém, mas não há diferença grande entre os modelos.

**Baseline da Etapa 1.** O notebook 01 registra F1 0,642 no teste, com seed 123, 17 features e `class_weight="balanced"`. Esse número vem de outra divisão e não é comparável com os desta tabela. A linha balanceada acima é a comparação válida.

Os números desta seção que não vêm do notebook 02, e os das seções 4.3 e 5, são gerados por `src/evaluate.py` e ficam em `models/02_avaliacao_campeao.json`.

### 4.2 Matriz de confusão (teste, limiar 0,5)

| | Previsto: permanece | Previsto: churn |
|---|---|---|
| **Real: permanece** | 787 | 248 |
| **Real: churn** | 83 | 291 |

- O modelo identifica **78%** dos clientes que cancelaram (291 de 374).
- Dos 539 clientes sinalizados, **54%** realmente cancelariam. Os outros **46%** (248) são alarmes falsos.
- **24%** dos clientes que permaneceram (248 de 1.035) são sinalizados indevidamente.
- 83 clientes que cancelaram passam despercebidos.

### 4.3 Métrica de negócio

Os clientes sinalizados concentram **80,5% da cobrança mensal** (`MonthlyCharges`) de todos os que cancelaram no conjunto de teste, com 38,3% da base sinalizada (539 de 1.409). Esta é a métrica definida no ML Canvas. Ela mede o alcance do modelo e **não representa dinheiro recuperado**, pois nenhuma ação de retenção foi simulada.

## 5. Possíveis vieses e desempenho por subgrupo

O modelo usa `gender`, `SeniorCitizen`, `Partner` e `Dependents`. Abaixo, o desempenho no conjunto de teste por grupo (limiar 0,5):

| Grupo | n | Churn real | Sinalizados | Recall | Precision | Alarmes falsos¹ | AUC |
|---|---|---|---|---|---|---|---|
| Mulheres | 687 | 28,1% | 37,6% | 74,6% | 55,8% | 23,1% | 0,850 |
| Homens | 722 | 25,1% | 38,9% | 81,2% | 52,3% | 24,8% | 0,855 |
| Menos de 65 anos | 1.187 | 23,3% | 33,2% | 73,6% | 51,5% | 21,0% | 0,855 |
| **65 anos ou mais** | 222 | 44,1% | **65,3%** | 89,8% | 60,7% | **46,0%** | **0,780** |
| Sem parceiro(a) | 736 | 33,7% | 51,1% | 82,7% | 54,5% | 35,0% | 0,820 |
| Com parceiro(a) | 673 | 18,7% | 24,2% | 68,3% | 52,8% | 14,1% | 0,874 |
| Sem dependentes | 1.083 | 32,0% | 47,4% | 81,0% | 54,8% | 31,5% | 0,825 |
| **Com dependentes** | 326 | 8,3% | 8,0% | **37,0%** | 38,5% | 5,4% | 0,888 |

¹ Percentual dos clientes do grupo que permaneceram e foram sinalizados como churn.

**O que a tabela mostra:**

- **Idosos (65+) são o ponto mais sensível.** O modelo sinaliza 65% deles, e 46% dos idosos que permaneceram recebem alarme falso, mais que o dobro dos 21% dos demais clientes. Parte disso reflete o churn real mais alto no grupo (44% contra 23%), mas o custo do erro fica mais concentrado neles, e o modelo ordena pior dentro desse grupo (AUC 0,78 contra 0,86). Se as ações de retenção forem ofertas, ligações ou descontos, esse grupo seria abordado com mais frequência do que o necessário.
- **Gênero:** as diferenças são pequenas (recall de 74,6% contra 81,2%). Com grupos desse tamanho, é provável que parte seja variação amostral.
- **Clientes com dependentes ou com muito tempo de casa são pouco detectados.** O recall com dependentes é de 37%. Na faixa de tenure acima de 48 meses, o recall é de 37% e a precision de 32%. O modelo tende a "não enxergar" o churn de perfis em que ele é raro.
- **Variáveis podem funcionar como proxy.** Retirar `gender` ou `SeniorCitizen` não elimina o problema, porque `Partner`, `Dependents`, `PaymentMethod` e `tenure` se correlacionam com características pessoais.
- **Não foi possível avaliar** vieses ligados a raça, etnia ou renda, porque o dataset não traz essas informações. A ausência de evidência aqui não significa ausência de viés.

**Cuidado ao interpretar:** os números vêm de um único conjunto de teste, sem intervalos de confiança. Os grupos pequenos são instáveis: há cerca de 27 clientes que cancelaram entre os 326 com dependentes e cerca de 35 entre os 450 com mais de 48 meses de tenure.

## 6. Limitações

- **Dados:** é um dataset público de exemplo, de um único estado dos EUA e de um único instante no tempo. Não há histórico mensal, então não foi possível validar o modelo ao longo do tempo (*out-of-time*). O desempenho em uma operadora brasileira pode ser bem diferente.
- **Calibração das probabilidades:** as pontuações de risco produzidas pelo modelo não devem ser interpretadas diretamente como probabilidades reais de cancelamento sem uma avaliação de calibração. Sua finalidade é auxiliar na identificação de clientes com maior propensão ao cancelamento.
- **Precisão do modelo:** o desempenho na identificação de clientes propensos ao cancelamento deve ser avaliado considerando métricas como precisão, recall e F1-score. A escolha do modelo deve levar em conta o equilíbrio entre identificar clientes em risco e evitar classificações incorretas.
- **Sem causalidade:** o modelo prevê quem tende a cancelar, mas não diz o que faria o cliente ficar. O efeito de uma ação de retenção exige um experimento próprio.
- **Sem monitoramento em produção:** não há detecção de drift nem retreino automático. O re-treino manual está em `src/train.py`.
- **Versões:** o arquivo `.joblib` foi gerado com scikit-learn 1.9.0 e deve ser carregado com a mesma versão. Como o formato é baseado em pickle, só carregue arquivos de origem confiável.
