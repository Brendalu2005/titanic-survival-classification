"""Dashboard local: python -m streamlit run streamlit_app.py."""

import pandas as pd
import streamlit as st

from src.dashboard_data import MODEL_NAMES, OUTCOMES, ROOT, filter_predictions, load_evaluation, load_predictions

st.set_page_config(page_title="Titanic | Previsões e avaliação", page_icon=":material/directions_boat:", layout="wide")
st.caption("TITANIC · MACHINE LEARNING PARA APRENDIZADO")
st.title("Previsões e desempenho")
st.markdown("Explore as previsões por passageiro e entenda como o modelo foi avaliado.")

with st.sidebar:
    st.header("Sobre este painel")
    st.caption("Projeto educacional. As previsões não têm finalidade operacional.")
    st.button("Atualizar dados", icon=":material/refresh:", width="stretch", key="refresh")
    st.caption("Após gerar novos relatórios ou previsões, atualize para ler os arquivos novamente.")

evaluation = None
try:
    evaluation = load_evaluation(ROOT / "reports")
except (OSError, ValueError) as exc:
    st.info(str(exc))

if evaluation is not None:
    selected = evaluation["selected_threshold"]
    threshold_label = f'{evaluation["threshold"]:.2f}'.replace(".", ",")
    model_label = MODEL_NAMES.get(evaluation["selected_model"], evaluation["selected_model"])
    test_rows = sum(sum(row) for row in evaluation["confusion_matrix"])
    st.caption(f"{model_label} · Limiar da avaliação: {threshold_label} · {test_rows} passageiros no teste com resposta conhecida")
    with st.container(horizontal=True):
        st.metric("Precisão", f'{selected["precision"]:.1%}'.replace(".", ","), border=True,
                  help="Entre os classificados como sobreviventes, quantos realmente sobreviveram.")
        st.metric("Recall", f'{selected["recall"]:.1%}'.replace(".", ","), border=True,
                  help="Entre os sobreviventes reais, quantos o modelo identificou.")
        st.metric("Acurácia", f'{selected["accuracy"]:.1%}'.replace(".", ","), border=True,
                  help="Proporção total de previsões corretas no conjunto de avaliação.")
    st.caption("Essas métricas vêm do teste separado do treino. Os filtros do lote de previsões não alteram esses valores.")

view = st.segmented_control("Visualização", ["Previsões", "Avaliação"], default="Previsões", key="view")

if view != "Avaliação":
    st.subheader("Previsões dos passageiros")
    st.caption("Resultados salvos do lote de previsão. Sem respostas reais, este lote não permite medir acurácia, precisão ou recall.")
    try:
        predictions = load_predictions(ROOT / "predictions/previsoes.csv", ROOT / "data/test.csv")
    except (OSError, ValueError) as exc:
        st.info(str(exc))
    else:
        with st.sidebar:
            st.divider()
            st.subheader("Filtrar previsões")
            outcomes = st.pills("Resultado previsto", list(OUTCOMES.values()), default=list(OUTCOMES.values()),
                                selection_mode="multi", key="outcomes")
            probability = st.slider("Probabilidade de sobrevivência (%)", 0, 100, (0, 100), step=5, key="probability")
            search = st.text_input("Buscar por nome ou ID", key="search", placeholder="Ex.: Kelly ou 892")
            classes = None
            if "Pclass" in predictions and predictions["Pclass"].notna().any():
                options = sorted(predictions["Pclass"].dropna().astype(int).unique().tolist())
                classes = st.pills("Classe da passagem", options, default=options, selection_mode="multi", key="classes")
            sexes = None
            if "Sex" in predictions and predictions["Sex"].notna().any():
                options = sorted(predictions["Sex"].dropna().unique().tolist())
                sexes = st.pills("Sexo", options, default=options, selection_mode="multi", key="sexes",
                                 format_func=lambda value: {"male": "Masculino", "female": "Feminino"}.get(value, value))
        filtered = filter_predictions(predictions, outcomes or [], probability, search, classes, sexes)
        st.markdown(f"**{len(filtered)} de {len(predictions)} passageiros** correspondem aos filtros.")
        if filtered.empty:
            st.info("Nenhum passageiro corresponde aos filtros. Altere a busca ou as seleções na barra lateral.")
        else:
            left, right = st.columns(2)
            with left:
                with st.container(border=True):
                    st.markdown("**Classes previstas neste recorte**")
                    counts = filtered["Previsao"].value_counts().reindex(OUTCOMES.values(), fill_value=0).rename("Passageiros")
                    st.bar_chart(counts, horizontal=True, height=230,
                                 alt="Quantidade de passageiros por classe prevista após os filtros")
            with right:
                with st.container(border=True):
                    st.markdown("**Distribuição das probabilidades**")
                    ranges = pd.cut(filtered["ProbabilidadeSobrevivencia"],
                                    bins=[0, .2, .4, .6, .8, 1], include_lowest=True,
                                    labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
                    histogram = ranges.value_counts(sort=False).rename("Passageiros")
                    st.bar_chart(histogram, height=230, alt="Distribuição das probabilidades previstas de sobrevivência")
            columns = [column for column in ["PassengerId", "Name", "Previsao", "ProbabilidadeSobrevivencia", "Age", "Sex", "Pclass"] if column in filtered]
            table = filtered[columns].rename(columns={
                "PassengerId": "ID", "Name": "Passageiro", "Previsao": "Previsão",
                "ProbabilidadeSobrevivencia": "Probabilidade", "Age": "Idade", "Sex": "Sexo", "Pclass": "Classe",
            })
            if "Sexo" in table:
                table["Sexo"] = table["Sexo"].replace({"male": "Masculino", "female": "Feminino"})
            st.dataframe(table, hide_index=True, height=380, alt="Previsões por passageiro, com probabilidade e características disponíveis",
                         column_config={"Probabilidade": st.column_config.ProgressColumn(
                             "Probabilidade de sobrevivência", min_value=0.0, max_value=1.0, format="percent")})
        # O download mantém o formato do CSV de previsão, inclusive quando o filtro fica vazio.
        csv = filtered[["PassengerId", "ProbabilidadeSobrevivencia", "Survived"]].to_csv(index=False).encode("utf-8")
        st.download_button("Baixar previsões filtradas", csv, file_name="previsoes_filtradas.csv",
                           mime="text/csv", icon=":material/download:", key="download", disabled=filtered.empty)
        st.caption("Sobreviveu / Não sobreviveu indicam a classe prevista, não a resposta histórica confirmada. As probabilidades e classes são lidas do CSV salvo.")

elif evaluation is not None:
    st.subheader("Avaliação com respostas conhecidas")
    st.info(evaluation.get("test_note", "O conjunto de teste já foi consultado no baseline. A avaliação é didática."))
    comparison = pd.DataFrame({
        "Dummy": evaluation["dummy"],
        "Modelo · limiar 0,50": evaluation["default_threshold"],
        f"Modelo · limiar {threshold_label} (escolhido no treino)": evaluation["selected_threshold"],
    }).T[["precision", "recall", "accuracy", "f1"]]
    comparison = comparison.rename(columns={"precision": "Precisão (%)", "recall": "Recall (%)",
                                            "accuracy": "Acurácia (%)", "f1": "F1 (%)"})
    st.dataframe((comparison * 100).round(1), alt="Comparação do baseline e dos limiares no mesmo conjunto de teste")
    st.caption("O limiar escolhido no treino é mantido mesmo quando 0,50 apresenta F1 maior no teste.")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Matriz de confusão · limiar escolhido**")
        matrix = pd.DataFrame(evaluation["confusion_matrix"],
                              index=["Real: não sobreviveu", "Real: sobreviveu"],
                              columns=["Previsto: não sobreviveu", "Previsto: sobreviveu"])
        st.table(matrix, alt="Contagens de acertos e erros para as duas classes no teste")
        st.caption("A diagonal contém os acertos; as outras células são os erros.")
    with right:
        with st.container(border=True):
            st.markdown("**Como interpretar as métricas**")
            st.markdown("**Precisão:** de quem foi previsto como sobrevivente, quantos sobreviveram.")
            st.markdown("**Recall:** de todos os sobreviventes reais, quantos foram identificados.")
            st.markdown("**Acurácia:** de todos os passageiros avaliados, quantos tiveram a classe prevista corretamente.")
            st.markdown("**F1:** combina precisão e recall; foi o critério usado para escolher o modelo.")
    with st.expander("Configuração do modelo"):
        st.json({"modelo": model_label, "limiar": evaluation["threshold"], "parametros": evaluation.get("parameters", {})})
else:
    st.info("Gere os relatórios com python -m src.train para visualizar a avaliação.")
