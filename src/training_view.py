"""Visualização didática dos relatórios de treinamento já concluído."""

import ast
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard_data import MODEL_NAMES
from src.dashboard_charts import f1_bars, f1_heatmap


def configuration_label(parameters):
    """Os CSVs da busca usam repr de dict; a comparação usa JSON."""
    parsed = ast.literal_eval(parameters)
    labels = {"C": "C", "max_depth": "Profundidade máxima",
              "min_samples_leaf": "Mínimo por folha", "n_estimators": "Árvores"}
    return "; ".join(
        f"{labels.get(key.removeprefix('model__'), key)}: "
        f"{'sem limite' if value is None else value}"
        for key, value in parsed.items()
    ) or "Classe mais frequente"


def load_training(reports):
    """Lê o resumo, preservando os valores calculados pelo treino."""
    try:
        comparison = pd.read_csv(reports / "comparacao_modelos.csv")
        required = ["model", "parameters", "train_f1", "cv_f1", "cv_f1_std"]
        if comparison.empty or not set(required).issubset(comparison.columns):
            raise ValueError("Resumo incompleto.")
        values = comparison[["train_f1", "cv_f1", "cv_f1_std"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Métricas inválidas.")
        if comparison["model"].duplicated().any():
            raise ValueError("Modelos repetidos.")
        comparison["Configuração"] = comparison["parameters"].map(configuration_label)
    except FileNotFoundError as exc:
        raise ValueError("O resumo do treinamento ainda não foi gerado. Execute python -m src.train.") from exc
    except (KeyError, TypeError, ValueError, SyntaxError) as exc:
        raise ValueError("Confira reports/comparacao_modelos.csv: o resumo está inválido.") from exc
    return comparison


def render_training(reports, evaluation):
    st.subheader("Entendendo o treinamento")
    st.caption("Resultados salvos do treinamento concluído mais recente em reports/. "
               "O painel não executa nem acompanha o treino ao vivo e não consulta o histórico do MLflow.")
    try:
        comparison = load_training(reports)
    except (OSError, ValueError) as exc:
        st.info(str(exc))
        return

    modified = datetime.fromtimestamp(
        (reports / "comparacao_modelos.csv").stat().st_mtime, ZoneInfo("America/Fortaleza")
    )
    st.caption(f"Resumo atualizado em {modified:%d/%m/%Y às %H:%M} (Fortaleza).")
    st.markdown("**Preparação → validação cruzada → escolha do modelo → ajuste do limiar → teste final**")
    st.markdown(
        "O CSV é separado em 80% para desenvolvimento e 20% para teste. "
        "Nos 80%, a validação cruzada estratificada usa cinco partes: em cada rodada, "
        "quatro treinam e uma valida. O pipeline aprende o preenchimento de ausentes, "
        "a codificação e, na regressão logística, a padronização apenas na parte de treino."
    )
    st.markdown("**F1 de treino × validação**")
    chart_view = st.segmented_control(
        "Tipo de gráfico", ["Ambos", "Heatmap", "Barras"], default="Ambos", key="training_chart_view")
    named = comparison.assign(Modelo=comparison["model"].map(lambda name: MODEL_NAMES.get(name, name)))
    chart_data = named[["Modelo", "train_f1", "cv_f1"]].rename(
        columns={"train_f1": "Treino", "cv_f1": "Validação"})
    if chart_view in ("Ambos", "Heatmap"):
        st.plotly_chart(f1_heatmap(chart_data), key="training_heatmap", width="stretch",
                        alt="Heatmap do F1 médio: modelos nas linhas, treino e validação nas colunas; valores de zero a um")
        st.caption("Cada linha representa um modelo. Quanto mais escura a célula, maior o F1. "
                   "Os valores nas células permitem comparar treino e validação diretamente; a escala de cor vai de 0 a 1.")
    if chart_view in ("Ambos", "Barras"):
        st.plotly_chart(f1_bars(chart_data, "Modelo"), key="training_bars", width="stretch",
                        alt="Barras agrupadas com F1 médio de treino e validação e valores sobre cada barra")
        st.caption("As duas visualizações mostram os mesmos resultados. Nas barras, compare a altura e os valores "
                   "de treino e validação de cada modelo.")
    table = named[["Modelo", "Configuração", "train_f1", "cv_f1", "cv_f1_std"]].rename(columns={
        "train_f1": "F1 médio · treino", "cv_f1": "F1 médio · validação", "cv_f1_std": "Desvio padrão · validação"})
    st.dataframe(table, hide_index=True, column_config={
        column: st.column_config.NumberColumn(format="%.3f") for column in table.columns[2:]
    }, alt="Melhores configurações, médias de F1 e variação entre as rodadas")
    st.caption("F1 combina precisão e recall para a classe sobreviveu; não é a porcentagem total de acertos. "
               "O desvio padrão mede a variação entre as rodadas; menor significa maior consistência nessas divisões.")
    candidates = comparison[comparison["model"] != "dummy"]
    if not candidates.empty:
        gap = candidates.assign(gap=candidates["train_f1"] - candidates["cv_f1"]).sort_values("gap").iloc[-1]
        if gap["gap"] > 0:
            st.info(f"{MODEL_NAMES.get(gap['model'], gap['model'])} apresentou a maior diferença entre treino "
                    f"({gap['train_f1']:.3f}) e validação ({gap['cv_f1']:.3f}). "
                    "Uma diferença maior é um sinal de possível sobreajuste: parte do aprendizado não se repete na validação.")
    if evaluation is not None:
        winner = MODEL_NAMES.get(evaluation["selected_model"], evaluation["selected_model"])
        st.markdown(f"**Modelo escolhido: {winner}.** O critério é o maior F1 médio na validação, "
                    "excluindo o Dummy. Diferenças pequenas devem ser interpretadas junto da variação entre as rodadas; "
                    "não comprovam superioridade estatística.")
    st.markdown("**Explorar as configurações e rodadas**")
    options = comparison["model"].tolist()
    selected_model = evaluation["selected_model"] if evaluation is not None else None
    model = st.selectbox("Modelo para estudar", options,
                         index=options.index(selected_model) if selected_model in options else 0,
                         format_func=lambda name: MODEL_NAMES.get(name, name), key="training_model")
    # Os nomes vêm exclusivamente da lista fixa, nunca de caminhos arbitrários no CSV.
    if model not in MODEL_NAMES:
        st.info("Não há relatório de busca disponível para este modelo.")
        return
    try:
        search = pd.read_csv(reports / f"busca_{model}.csv")
        search["Configuração"] = search["params"].map(configuration_label)
        st.dataframe(search[["Configuração", "mean_train_f1", "mean_test_f1", "std_test_f1"]].rename(columns={
            "mean_train_f1": "F1 médio · treino", "mean_test_f1": "F1 médio · validação",
            "std_test_f1": "Desvio padrão · validação"}).round(3), hide_index=True,
            alt="Todas as configurações avaliadas para o modelo selecionado")
        best = search.loc[search["mean_test_f1"].idxmax()]
        summary = comparison.loc[comparison["model"] == model].iloc[0]
        if not np.isclose(best["mean_test_f1"], summary["cv_f1"]):
            raise ValueError("Relatórios de execuções diferentes.")
        folds = pd.DataFrame({
            "Rodada": [str(i + 1) for i in range(5)],
            "Treino": [best[f"split{i}_train_f1"] for i in range(5)],
            "Validação": [best[f"split{i}_test_f1"] for i in range(5)],
        })
        st.markdown(f"**Rodadas da melhor configuração:** {best['Configuração']}")
        st.plotly_chart(f1_bars(folds, "Rodada"), key="training_folds", width="stretch",
                        alt="F1 de treino e validação nas cinco rodadas da melhor configuração")
        st.caption("As rodadas usam partes diferentes dos dados; não são etapas de melhoria contínua. "
                   "Nas buscas do scikit-learn, as colunas test significam validação interna, não o teste final de 20%.")
        if model == "xgboost":
            st.caption("Na busca do XGBoost, a taxa de aprendizado é fixa em 0,1; variam profundidade e quantidade de árvores.")
    except (OSError, KeyError, TypeError, ValueError, SyntaxError):
        st.info("Os detalhes da busca estão ausentes, inválidos ou não correspondem ao resumo. "
                "Execute python -m src.train para gerar os relatórios completos.")

    if evaluation is not None:
        st.markdown("**Teste final · matriz de confusão**")
        st.markdown("Depois da escolha do modelo, o limiar é ajustado com previsões fora da partição nos dados "
                    "de desenvolvimento. Só então o modelo é avaliado nos 20% de teste.")
        st.table(pd.DataFrame(evaluation["confusion_matrix"],
                             index=["Real: não sobreviveu", "Real: sobreviveu"],
                             columns=["Previsto: não sobreviveu", "Previsto: sobreviveu"]),
                 alt="Acertos e erros no teste final com o limiar escolhido no treino")
        st.caption("A diagonal contém os acertos; as outras células são os erros. "
                   "A seção Avaliação detalha as métricas e os limiares.")
        st.info(evaluation.get("test_note", "O teste já foi consultado no baseline; a avaliação é didática."))
    else:
        st.info("A avaliação final ainda não está disponível. Execute python -m src.train.")
