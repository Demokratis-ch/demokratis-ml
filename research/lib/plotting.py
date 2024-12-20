from collections.abc import Callable
from typing import Any

import matplotlib.figure
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics


def plot_and_log(plot_function: Callable, mlflow_file_name: str, **plot_kwargs: Any) -> matplotlib.figure.Figure:
    """Plot and log the figure to MLFlow."""
    fig, ax = plt.subplots()
    plot_function(ax=ax, **plot_kwargs)
    mlflow.log_figure(fig, mlflow_file_name)
    plt.close(fig)
    return fig


def plot_classification_report_heatmap(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    target_names: list[str],
) -> matplotlib.figure.Figure:
    """Colorize the standard sklearn classification report like a heatmap."""
    report = sklearn.metrics.classification_report(
        ground_truth, predictions, target_names=target_names, output_dict=True
    )
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(5, 8), width_ratios=[3, 1])
    fig.subplots_adjust(wspace=0.01)

    df_metrics = pd.DataFrame(report).iloc[:-1, :].transpose()
    df_support = pd.DataFrame(report).transpose()[["support"]]
    # Remove the support totals so that the color scale is not affected by them.
    for total_cell in ("micro avg", "macro avg", "weighted avg", "samples avg"):
        df_support.loc[total_cell] = np.nan

    sns.heatmap(df_metrics, cmap="coolwarm", ax=ax1, cbar=False, annot=True, fmt=".2f")
    sns.heatmap(df_support, cmap="coolwarm", ax=ax2, cbar=False, annot=True, fmt=".0f")
    ax2.set_yticks([])

    fig.subplots_adjust(wspace=0.001)
    plt.close(fig)
    return fig


def plot_score_against_support(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    target_names: list[str],
    score_metric: str = "f1-score",
    ylim: tuple[float, float] = (0.0, 1.0),
) -> matplotlib.figure.Figure:
    """Plot a scatter plot of the score metric against the support for each class."""
    report = pd.DataFrame(
        sklearn.metrics.classification_report(ground_truth, predictions, target_names=target_names, output_dict=True)
    ).transpose()
    report = report[["support", score_metric]]
    report = report.drop(["micro avg", "macro avg", "weighted avg", "samples avg"])
    report = report.sort_values(by="support")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x="support", y=score_metric, data=report, ax=ax)
    for i, txt in enumerate(report.index):
        ax.annotate(
            txt.replace("topic_", ""),
            (report["support"].iloc[i], report[score_metric].iloc[i]),
            fontsize=8,
        )
    plt.ylim(*ylim)
    plt.close(fig)
    return fig
