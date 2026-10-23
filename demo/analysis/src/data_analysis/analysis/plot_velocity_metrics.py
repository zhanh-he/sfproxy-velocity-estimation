import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 你最常改的地方
# =========================
FIGSIZE = (8.5, 4.8)   # (宽, 高)
DPI = 220
SAVE_DIR = "."         # 图片保存目录
SAVE_PNG = True
SHOW_LEGEND = True
SHOW_BAR_LABELS = True
SHOW_TITLE = True

COLORS = {
    "Ground truth": "#4C78A8",
    "Flat 64": "#F58518",
    "Flat 80": "#54A24B",
    "Flat 64 minus GT": "#4C78A8",
    "Flat 80 minus GT": "#F58518",
    "GT - Flat 64": "#4C78A8",
    "GT - Flat 80": "#F58518",
    "Avg |Sal - YDP|": "#54A24B",
}

# =========================
# 数据（跟前面预览图一致）
# =========================
sal = {
    "pearson": {"BSSL": 0.8700, "BSTL": 0.8829, "PHE": 0.7743, "OSF": 0.9029},
    "cosine":  {"BSSL": 0.9388, "BSTL": 0.9748, "PHE": 0.9867, "OSF": 0.9724},
    "mae":     {"BSSL": 0.7024, "BSTL": 3.7058, "PHE": 0.9293, "OSF": 0.0309},
}

flat64_sal = {
    "pearson": {"BSSL": 0.7519, "BSTL": 0.6199, "PHE": 0.4114, "OSF": 0.5152},
    "cosine":  {"BSSL": 0.8870, "BSTL": 0.9275, "PHE": 0.9677, "OSF": 0.8754},
    "mae":     {"BSSL": 0.7858, "BSTL": 4.0647, "PHE": 1.0440, "OSF": 0.0341},
}

flat80_sal = {
    "pearson": {"BSSL": 0.7467, "BSTL": 0.6163, "PHE": 0.4114, "OSF": 0.5172},
    "cosine":  {"BSSL": 0.8864, "BSTL": 0.9267, "PHE": 0.9717, "OSF": 0.8750},
    "mae":     {"BSSL": 0.6774, "BSTL": 3.3035, "PHE": 0.7970, "OSF": 0.0291},
}

ydp = {
    "pearson": {"BSSL": 0.8363, "BSTL": 0.8724, "PHE": 0.8201, "OSF": 0.8857},
}

flat64_ydp = {
    "pearson": {"BSSL": 0.7329, "BSTL": 0.6168, "PHE": 0.4250, "OSF": 0.4964},
}

flat80_ydp = {
    "pearson": {"BSSL": 0.7205, "BSTL": 0.6196, "PHE": 0.4212, "OSF": 0.4976},
}


def maybe_save(filename):
    if SAVE_PNG:
        os.makedirs(SAVE_DIR, exist_ok=True)
        plt.savefig(os.path.join(SAVE_DIR, filename), dpi=DPI, bbox_inches="tight")


def maybe_bar_label(bars):
    if SHOW_BAR_LABELS:
        plt.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)


def maybe_title(title):
    if SHOW_TITLE:
        plt.title(title, fontsize=13)


def maybe_legend():
    if SHOW_LEGEND:
        plt.legend()


def grouped_bar(categories, series_dict, title, ylabel, ylim=None, filename=None, note=None):
    """最简单的 grouped bar 封装。"""
    plt.figure(figsize=FIGSIZE)
    x = np.arange(len(categories))
    n_series = len(series_dict)
    width = 0.8 / n_series
    offsets = np.linspace(-(n_series - 1) / 2 * width, (n_series - 1) / 2 * width, n_series)

    for i, (label, values) in enumerate(series_dict.items()):
        bars = plt.bar(
            x + offsets[i],
            values,
            width=width,
            label=label,
            color=COLORS.get(label, None),
        )
        maybe_bar_label(bars)

    plt.xticks(x, categories, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    maybe_title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    maybe_legend()
    if note is not None:
        plt.figtext(0.5, -0.02, note, ha="center", fontsize=9)
    plt.tight_layout()
    if filename is not None:
        maybe_save(filename)
    plt.show()


# =========================
# Figure 1
# Salamander: Pearson on BSSL/BSTL
# =========================
def plot_fig1():
    categories = ["BSSL", "BSTL"]
    series = {
        "Ground truth": [sal["pearson"][c] for c in categories],
        "Flat 64": [flat64_sal["pearson"][c] for c in categories],
        "Flat 80": [flat80_sal["pearson"][c] for c in categories],
    }
    grouped_bar(
        categories=categories,
        series_dict=series,
        title="Salamander: Pearson r cleanly separates GT from flat baselines",
        ylabel="Pearson r",
        ylim=(0.55, 0.95),
        filename="fig1_salamander_pearson_primary.png",
    )


# =========================
# Figure 2
# Salamander: cosine similarity on all 4 features
# =========================
def plot_fig2():
    categories = ["BSSL", "BSTL", "PHE", "OSF"]
    series = {
        "Ground truth": [sal["cosine"][c] for c in categories],
        "Flat 64": [flat64_sal["cosine"][c] for c in categories],
        "Flat 80": [flat80_sal["cosine"][c] for c in categories],
    }
    grouped_bar(
        categories=categories,
        series_dict=series,
        title="Salamander: cosine similarity remains high across all settings",
        ylabel="Cosine similarity",
        ylim=(0.84, 1.00),
        filename="fig2_salamander_cosine_ceiling.png",
        note="All bars sit near the ceiling; GT-vs-flat differences are visually compressed.",
    )


# =========================
# Figure 3
# Salamander: MAE difference (Flat minus GT)
# 正值 = flat 比 GT 更差
# 负值 = flat 比 GT 更好（就是你想强调的 mis-ranking）
# =========================
def plot_fig3():
    categories = ["BSSL", "BSTL", "PHE", "OSF"]
    flat64_minus_gt = [flat64_sal["mae"][c] - sal["mae"][c] for c in categories]
    flat80_minus_gt = [flat80_sal["mae"][c] - sal["mae"][c] for c in categories]

    plt.figure(figsize=FIGSIZE)
    x = np.arange(len(categories))
    width = 0.32

    bars1 = plt.bar(
        x - width / 2,
        flat64_minus_gt,
        width=width,
        label="Flat 64 minus GT",
        color=COLORS["Flat 64 minus GT"],
    )
    bars2 = plt.bar(
        x + width / 2,
        flat80_minus_gt,
        width=width,
        label="Flat 80 minus GT",
        color=COLORS["Flat 80 minus GT"],
    )

    plt.axhline(0, color="black", linewidth=1)
    maybe_bar_label(bars1)
    maybe_bar_label(bars2)
    plt.xticks(x, categories, fontsize=11)
    plt.ylabel("MAE difference", fontsize=11)
    maybe_title("Salamander: MAE can invert the expected ranking")
    maybe_legend()
    plt.figtext(
        0.5,
        -0.02,
        "Positive = flat baseline is worse than GT. Negative = flat baseline scores better than GT on MAE.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout()
    maybe_save("fig3_salamander_mae_misranking.png")
    plt.show()


# =========================
# Figure 4
# Pearson: GT-vs-flat gap vs cross-instrument gap
# =========================
def plot_fig4():
    categories = ["BSSL", "BSTL"]

    gt_minus_flat64 = [sal["pearson"][c] - flat64_sal["pearson"][c] for c in categories]
    gt_minus_flat80 = [sal["pearson"][c] - flat80_sal["pearson"][c] for c in categories]
    avg_instr_gap = [
        np.mean([
            abs(sal["pearson"][c] - ydp["pearson"][c]),
            abs(flat64_sal["pearson"][c] - flat64_ydp["pearson"][c]),
            abs(flat80_sal["pearson"][c] - flat80_ydp["pearson"][c]),
        ])
        for c in categories
    ]

    series = {
        "GT - Flat 64": gt_minus_flat64,
        "GT - Flat 80": gt_minus_flat80,
        "Avg |Sal - YDP|": avg_instr_gap,
    }

    grouped_bar(
        categories=categories,
        series_dict=series,
        title="Pearson on BSSL/BSTL: sensitivity is much larger than instrument gap",
        ylabel="Magnitude",
        ylim=(0.0, 0.32),
        filename="fig4_pearson_gain_vs_instrument_gap.png",
        note="A good primary metric should show a large GT-vs-flat gap and a small cross-instrument gap.",
    )


if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
