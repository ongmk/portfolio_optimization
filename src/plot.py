from backtrader_plotly.plotter import BacktraderPlotly
from backtrader_plotly.scheme import PlotScheme


def plot_with_plotly(cerebro):
    scheme = PlotScheme(decimal_places=5, max_legend_text_width=16)
    figs = cerebro.plot(BacktraderPlotly(show=False, scheme=scheme))

    for i, each_run in enumerate(figs):
        for j, each_strategy_fig in enumerate(each_run):
            # open plot in browser
            each_strategy_fig.show()
    pass
