import plotly.io
from backtrader_plotly.plotter import BacktraderPlotly
from backtrader_plotly.scheme import PlotScheme


def plot_with_plotly(cerebro):
    scheme = PlotScheme(decimal_places=5, max_legend_text_width=16)
    figs = cerebro.plot(BacktraderPlotly(show=False, scheme=scheme))

    for i, each_run in enumerate(figs):
        for j, each_strategy_fig in enumerate(each_run):
            each_strategy_fig.show()
            html = plotly.io.to_html(each_strategy_fig, full_html=False)

            plotly.io.write_html(
                each_strategy_fig, f"./data/{i}_{j}.html", full_html=True
            )

    pass
