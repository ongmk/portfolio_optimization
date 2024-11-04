from backtrader import Observer


class ClassAttrObserver(Observer):
    lines = ("value",)
    plotinfo = dict(plot=True, subplot=True)

    params = dict(attr=None, name=None)

    def __init__(self):
        assert self.p.attr is not None, "Attribute must be set"
        self.plotinfo.plotname = self.p.name or self.p.attr

    def next(self):
        self.lines.value[0] = getattr(self._owner, self.p.attr)


class WeightObserver(Observer):
    params = dict(ticker=None)
    lines = ("weight",)
    plotinfo = dict(plot=True, subplot=True)

    def __init__(self):
        self.plotinfo.plotname = self.p.ticker

    def next(self):
        self.lines.weight[0] = self._owner.weights[self.p.ticker]
