from bisect import bisect_left
from collections import defaultdict

import matplotlib.dates as mdates


class EventViewer:
    def __init__(self, fig, price_axis, activity_axis, dates, values, events):
        self.fig = fig
        self.price_axis = price_axis
        self.activity_axis = activity_axis
        self.dates = dates
        self.values = values
        self.events = events
        self.page = 0
        self.page_size = 20
        self.selected_day = None
        self.labels = []
        self.panel = fig.add_axes([0.73, 0.16, 0.26, 0.65])
        self.panel.axis('off')
        self.body = self.panel.text(0, 1, '', va='top', fontsize=9, family='monospace')
        self.header = fig.text(0.73, 0.85, 'Membership events', fontsize=11)
        fig.text(0.73, 0.07, 'Click: nearest event date\nArrows: pages | Escape: visible range', fontsize=8)
        price_axis.callbacks.connect('xlim_changed', self.range_changed)
        fig.canvas.mpl_connect('button_press_event', self.clicked)
        fig.canvas.mpl_connect('key_press_event', self.key_pressed)
        self.refresh()

    def visible_events(self):
        left, right = sorted(self.price_axis.get_xlim())
        return [event for event in self.events if left <= mdates.date2num(event[0]) <= right]

    def refresh(self):
        visible = self.visible_events()
        rows = [event for event in visible if self.selected_day is None or event[0] == self.selected_day]
        pages = max(1, (len(rows) + self.page_size - 1) // self.page_size)
        self.page = min(self.page, pages - 1)
        text = [f'{len(rows)} events | page {self.page + 1}/{pages}', '']
        for day, ticker, action in rows[self.page*self.page_size:(self.page+1)*self.page_size]:
            sign = '+' if action == 'Added' else '-'
            text.append(f'{day:%Y-%m-%d} {sign} {ticker}')
        if not rows:
            text.append('No events in view')
        self.body.set_text('\n'.join(text))
        for label in self.labels:
            label.remove()
        self.labels.clear()
        grouped = defaultdict(list)
        for day, ticker, action in visible:
            grouped[day].append(('+' if action == 'Added' else '-') + ticker)
        if len(visible) <= 16:
            for i, (day, symbols) in enumerate(sorted(grouped.items())):
                position = bisect_left(self.dates, day)
                if position >= len(self.dates):
                    continue
                label = self.price_axis.annotate(
                    f'{day:%Y-%m-%d}\n' + ', '.join(symbols),
                    (self.dates[position], self.values[position]),
                    xytext=(0, 25 + 30 * (i % 3)), textcoords='offset points',
                    ha='center', fontsize=8,
                    arrowprops={'arrowstyle': '-', 'color': '0.5'},
                    bbox={'facecolor': 'white', 'edgecolor': '0.8', 'alpha': .9},
                )
                self.labels.append(label)
        self.fig.canvas.draw_idle()

    def range_changed(self, axis):
        self.selected_day = None
        self.page = 0
        self.refresh()

    def clicked(self, event):
        if event.inaxes not in (self.price_axis, self.activity_axis) or event.xdata is None:
            return
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        if toolbar is not None and toolbar.mode:
            return
        visible = self.visible_events()
        if visible:
            self.selected_day = min(visible, key=lambda row: abs(mdates.date2num(row[0])-event.xdata))[0]
            self.page = 0
            self.refresh()

    def key_pressed(self, event):
        if event.key == 'right':
            self.page += 1
        elif event.key == 'left':
            self.page = max(0, self.page - 1)
        elif event.key == 'escape':
            self.selected_day = None
            self.page = 0
        else:
            return
        self.refresh()
