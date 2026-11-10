"""Figuras Plotly Express compartilhadas pelo dashboard."""

import plotly.express as px


def f1_bars(data, category):
    """Compara treino e validação com a mesma escala e rótulos legíveis."""
    long = data.melt(id_vars=category, value_vars=["Treino", "Validação"],
                     var_name="Conjunto", value_name="F1")
    long["Valor"] = long["F1"].map(lambda value: f"{value:.3f}".replace(".", ","))
    figure = px.bar(long, x=category, y="F1", color="Conjunto", barmode="group", text="Valor",
                    category_orders={category: data[category].tolist(), "Conjunto": ["Treino", "Validação"]},
                    labels={"F1": "F1 médio" if category == "Modelo" else "F1"}, height=350)
    figure.update_traces(textposition="outside", cliponaxis=False,
                         hovertemplate="%{x}<br>F1: %{y:.3f}<extra>%{fullData.name}</extra>")
    figure.update_layout(legend=dict(title=None, orientation="h", y=1.12),
                         margin=dict(t=45, b=35), separators=",.")
    figure.update_xaxes(title=None, tickangle=0, type="category")
    figure.update_yaxes(range=[0, 1.08], tickvals=[0, .2, .4, .6, .8, 1])
    return figure


def f1_heatmap(data):
    """Mantém a escala 0–1 e os valores visíveis, incluindo o zero do Dummy."""
    matrix = data.set_index("Modelo")[["Treino", "Validação"]]
    figure = px.imshow(matrix, text_auto=".3f", zmin=0, zmax=1, aspect="auto",
                       color_continuous_scale=["#F2F6FA", "#087E8B"],
                       labels=dict(x="Conjunto", y="Modelo", color="F1"),
                       height=130 + 54 * len(matrix))
    figure.update_traces(xgap=2, ygap=2,
                         hovertemplate="%{y}<br>%{x}<br>F1: %{z:.3f}<extra></extra>")
    figure.update_layout(margin=dict(t=35, b=20), separators=",.")
    figure.update_xaxes(side="top", title=None, tickangle=0)
    figure.update_yaxes(title=None)
    return figure
