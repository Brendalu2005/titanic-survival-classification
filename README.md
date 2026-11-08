# Titanic — classificação de sobrevivência

Projeto **exclusivamente educacional** para praticar um fluxo de Machine Learning.
Usa dados históricos para estudar associações e previsão; não serve para decisões reais.

## Como rodar
Execute os comandos na raiz do repositório, em PowerShell.
Ambiente validado: Python 3.14.5 em Windows, sem GPU.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se a `.venv` já existir, apenas ative-a.
Caso a ativação esteja bloqueada, use `.\.venv\Scripts\python.exe` no lugar de `python`.

Baixe [train.csv do Titanic no Kaggle](https://www.kaggle.com/competitions/titanic/data)
(pode exigir login e aceite das regras) e coloque em `data/train.csv`.
Para prever passageiros sem rótulos, baixe também `test.csv` para `data/test.csv`.

```powershell
python -m src.train
python -m src.predict --input data/test.csv
python -m unittest discover -s tests -v
python -m jupyterlab #só outra forma de abrir os notebooks sem ser via vscode
```

Leia os notebooks na ordem: **01 baseline → 02 EDA → 03 modelos**.
O novo notebook também executa o treinamento: rodá-lo depois do terminal repete
o experimento e atualiza os relatórios; as execuções anteriores permanecem no MLflow.
Os dois notebooks originais foram preservados. A EDA ainda tem interpretações
e conclusões em branco a serem preenchidas.

## Fluxo
```mermaid
flowchart TD
    A[CSV local do Kaggle] --> B[Baseline e EDA]
    B --> C[Divisão estratificada 80/20 - semente 42]
    C --> D[Features e preparação dentro dos pipelines]
    D --> E[Validação cruzada e busca pequena no treino]
    E --> F[Escolha do modelo e limiar no treino]
    F --> G[Avaliação didática nos 20%]
    G --> H[Salvar pipeline, limiar e relatórios]
    H --> I[Previsões em outro CSV]
    I --> K[Dashboard Streamlit local]
    H --> K
    E --> J[MLflow local]
    G --> J
```

Compara dummy, regressão logística, árvore e XGBoost em CPU.
Usa F1 dos sobreviventes como critério didático, com acurácia, precisão, recall
e ROC-AUC de apoio. Ajusta o limiar com previsões fora da partição de treino.
O teste já foi consultado no baseline: a avaliação **não é independente**.
A seleção interna também pode ser otimista; não usamos validação cruzada aninhada.
Dados ausentes e passageiros da mesma família são outras limitações.

## Resultados e experimentos
- `src/`: funções compartilhadas de preparação, treinamento e previsão.
- `reports/`: comparações, métricas, gráficos, metadados e conclusões em cinco linhas.
- `models/titanic.joblib`: pipeline, limiar e colunas esperadas, ajustados só nos 80%.
- `predictions/previsoes.csv`: identificador, probabilidade e classe prevista.
- `mlflow.db` e `mlartifacts/`: histórico local de experimentos e artefatos.

Para consultar o MLflow, após treinar:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abra http://127.0.0.1:5000 e selecione `titanic-aprendizado`; encerre com Ctrl+C.
As buscas registram parâmetros e métricas; a avaliação registra relatórios e modelo.
Dados, modelo, previsões e armazenamento do MLflow ficam fora do Git.
Treino ou previsão com CSV ausente ou colunas faltantes informa o problema no terminal.
O dashboard é local e educacional, sem deploy ou uso operacional.

## Dashboard
Após treinar e gerar as previsões, execute:

```powershell
python -m streamlit run streamlit_app.py
```

Abra o endereço local mostrado no terminal. O painel lê os relatórios e o CSV salvo,
permite filtrar passageiros e baixar previsões. Precisão, recall e acurácia vêm
do teste com respostas conhecidas; os filtros das previsões não recalculam métricas.
