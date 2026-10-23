import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")

from data_analysis.analysis import plot_velocity_metrics as pvm


class PlotVelocityMetricsTests(unittest.TestCase):
    @mock.patch.object(pvm.plt, "show")
    @mock.patch.object(pvm.plt, "legend")
    @mock.patch.object(pvm.plt, "title")
    @mock.patch.object(pvm.plt, "bar_label")
    def test_grouped_bar_can_hide_labels_legend_and_title(
        self,
        bar_label_mock,
        title_mock,
        legend_mock,
        show_mock,
    ):
        pvm.SHOW_BAR_LABELS = False
        pvm.SHOW_LEGEND = False
        pvm.SHOW_TITLE = False

        pvm.grouped_bar(
            categories=["A", "B"],
            series_dict={"Ground truth": [0.1, 0.2], "Flat 64": [0.2, 0.3]},
            title="demo title",
            ylabel="demo ylabel",
        )

        bar_label_mock.assert_not_called()
        title_mock.assert_not_called()
        legend_mock.assert_not_called()
        show_mock.assert_called_once()

    @mock.patch.object(pvm.plt, "show")
    @mock.patch.object(pvm.plt, "legend")
    @mock.patch.object(pvm.plt, "title")
    @mock.patch.object(pvm.plt, "bar_label")
    def test_plot_fig3_can_hide_labels_legend_and_title(
        self,
        bar_label_mock,
        title_mock,
        legend_mock,
        show_mock,
    ):
        pvm.SHOW_BAR_LABELS = False
        pvm.SHOW_LEGEND = False
        pvm.SHOW_TITLE = False

        pvm.plot_fig3()

        bar_label_mock.assert_not_called()
        title_mock.assert_not_called()
        legend_mock.assert_not_called()
        show_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
