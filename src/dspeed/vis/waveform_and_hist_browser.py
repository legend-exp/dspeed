from __future__ import annotations

import itertools
import logging

import lgdo
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .waveform_browser import WaveformBrowser

log = logging.getLogger(__name__)


class WaveformAndHistBrowser(WaveformBrowser):
    """
    The :class:`WaveformAndHistBrowser` extends :class:`WaveformBrowser` to provide
    interactive browsing and visualization of histograms in addition to waveforms.
    It supports drawing waveforms, multiple histograms (with custom styles), and offers options
    for vertical/horizontal orientation and logarithmic axes. Histogram data is specified via
    value/edge pairs, and can be visualized alongside or instead of waveforms.
    """

    def __init__(
        self,
        *args,
        hist_values_edges: tuple[str, str] | list[tuple[str, str]],
        hist_styles: list[dict[str, list]] | None = None,
        vertical_hist: bool = False,
        hist_log: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        hist_values_edges
            Tuple or list of tuples specifying the names of histogram values and edges,
            which are defined in the dsp_config (see :class:`WaveformBrowser`)
        hist_styles
            List of style dictionaries for histograms. Each dictionary should map style properties to lists of values.
        vertical_hist
            If ``True``, draw histograms vertically (only allowed if no lines are drawn).
        hist_log
            If ``True``, use logarithmic scale for histogram counts.
        args, kwargs
            Additional (keyword) arguments passed to :class:`WaveformBrowser`.
        """
        self.values_edges_names = (
            hist_values_edges
            if isinstance(hist_values_edges, list)
            else [hist_values_edges]
        )
        super().__init__(
            *args,
            additional_outputs=[x for y in self.values_edges_names for x in y],
            **kwargs,
        )
        self.values_edges_data = [([], [])] * len(self.values_edges_names)

        self.hist_styles = [None] * len(self.values_edges_names)
        if hist_styles is not None:
            assert isinstance(hist_styles, list)
            for i, sty in enumerate(hist_styles):
                if sty is None:
                    self.hist_styles[i] = None
                else:
                    self.hist_styles[i] = itertools.cycle(cycler(**sty))

        if vertical_hist and len(self.lines) > 0:
            raise RuntimeError(
                "Cannot draw vertical histograms when also "
                "drawing waveforms. Use lines=[] in this case."
            )
        self.vertical_hist = vertical_hist
        self.hist_log = hist_log

    def new_figure(self, *args, **kwargs) -> None:
        """
        Create a new figure and axis for drawing waveforms and histograms.
        If vertical histograms are not requested, create a secondary xaxis for histograms.
        """
        super().new_figure(*args, **kwargs)
        if not self.vertical_hist:
            self.ax2 = self.ax.twiny()

    def set_figure(self, fig: WaveformBrowser | Figure, ax: Axes = None) -> None:
        """
        Use an existing figure and axis for drawing.
        If vertical histograms are not requested, create a secondary axis for histograms.

        Parameters
        ----------
        fig
            Existing :class:`WaveformBrowser` or :class:`matplotlib.figure.Figure` to use.
        ax
            Existing :class:`matplotlib.axes.Axes` to use (optional).
        """
        super().set_figure(fig, ax)
        if not self.vertical_hist:
            self.ax2 = self.ax.twiny()

    def clear_data(self) -> None:
        """
        Reset the currently stored data.
        Derived class data is reset before base class data.
        """
        for val_edg in self.values_edges_data:
            val_edg[0].clear()
            val_edg[1].clear()
        super().clear_data()

    def find_entry(self, entry: int | list[int], *args, **kwargs) -> None:
        """
        Find the requested entry or entries and store associated waveform and histogram data internally.
        For each entry, extract histogram values and edges and store them for later drawing.

        Parameters
        ----------
        entry
            Index or list of indices to find.
        args, kwargs
            Additional arguments passed to base class.
        """
        super().find_entry(entry, *args, **kwargs)
        if hasattr(entry, "__iter__"):
            # super().find_entry() recurses in this case
            return
        assert isinstance(entry, int)
        i_tb = entry - self.lh5_it.current_i_entry
        assert len(self.lh5_out) > i_tb >= 0
        for i, (val_n, edg_n) in enumerate(self.values_edges_names):
            val_data = self.lh5_out.get(val_n, None)
            edg_data = self.lh5_out.get(edg_n, None)
            if not isinstance(val_data, lgdo.ArrayOfEqualSizedArrays):
                raise RuntimeError(
                    f"histogram values {val_n} has to be instance of lgdo.ArrayOfEqualSizedArrays"
                )
            if not isinstance(edg_data, lgdo.ArrayOfEqualSizedArrays):
                raise RuntimeError(
                    f"histogram edges {edg_n} has to be instance of lgdo.ArrayOfEqualSizedArrays"
                )
            self.values_edges_data[i][0].append(val_data.view_as("ak")[i_tb].to_numpy())
            self.values_edges_data[i][1].append(edg_data.view_as("ak")[i_tb].to_numpy())

    def draw_current(self, clear: bool = True, *args, **kwargs) -> None:
        """
        Draw the currently stored waveforms and histograms in the figure.
        If waveforms are present, draw them using the base class and draw histograms on a secondary axis.
        If only histograms are present, draw them on the main axis, optionally vertically.

        Parameters
        ----------
        clear
            If ``True``, clear the axes before drawing.
        args, kwargs
            Additional arguments passed to base class.
        """
        use_ax = None
        orientation = "horizontal"
        if len(self.lines) > 0:
            super().draw_current(clear, *args, **kwargs)
            if clear:
                self.ax2.clear()
            self.ax2.set_ylim(self.ax.get_ylim())
            use_ax = self.ax2
        else:
            # No lines drawn by base class; only histograms requested
            if not (self.ax and self.fig and plt.fignum_exists(self.fig.number)):
                self.new_figure()
            use_ax = self.ax
            if self.vertical_hist:
                orientation = "vertical"
        assert use_ax is not None

        if self.hist_log:
            if self.vertical_hist:
                use_ax.set_yscale("log")
            else:
                use_ax.set_xscale("log")

        default_style = itertools.cycle(cycler(plt.rcParams["axes.prop_cycle"]))
        for i, (values_list, edges_list) in enumerate(self.values_edges_data):
            styles = self.hist_styles[i]
            if styles is None:
                styles = default_style
            else:
                styles = iter(styles)
            for values, edges in zip(values_list, edges_list):
                sty = next(styles)
                use_ax.stairs(values, edges, orientation=orientation, **sty)
