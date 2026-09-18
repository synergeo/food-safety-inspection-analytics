"""
Reusable visualization utilities for the BVL Food Safety Inspection
Analytics project.

The functions in this module reproduce the main visualization patterns
used during exploratory data analysis in the final project notebook.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_target_distribution(model_df):
    """
    Plot the distribution of non-adverse and adverse food samples.
    """
    target_counts = (
        model_df["Target_Adverse"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(7, 5))

    ax = sns.barplot(
        x=["Non-Adverse", "Adverse"],
        y=target_counts.values,
    )

    plt.title("Distribution of Food-Sample Outcomes")
    plt.xlabel("Sample Outcome")
    plt.ylabel("Number of Samples")

    for i, value in enumerate(target_counts.values):
        ax.text(
            i,
            value + 80,
            f"{value:,}",
            ha="center",
            fontweight="bold",
        )

    plt.tight_layout()
    return ax


def plot_category_distribution(
    df,
    category,
    title,
    ylabel=None,
):
    """
    Plot the number of samples belonging to each category.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample-level dataset.

    category : str
        Categorical column to summarize.

    title : str
        Chart title.

    ylabel : str, optional
        Human-readable y-axis label.
    """
    counts = (
        df[category]
        .value_counts()
        .sort_values(ascending=True)
    )

    plt.figure(figsize=(11, 8))

    ax = sns.barplot(
        x=counts.values,
        y=counts.index,
    )

    plt.title(title)
    plt.xlabel("Number of Samples")
    plt.ylabel(ylabel or category)

    for i, value in enumerate(counts.values):
        ax.text(
            value,
            i,
            f" {value:,}",
            va="center",
        )

    plt.tight_layout()
    return ax


def calculate_adverse_rate(
    df,
    category,
    min_samples=0,
):
    """
    Calculate observed adverse rate by category.

    Parameters
    ----------
    df : pandas.DataFrame
        Sample-level dataset containing Target_Adverse.

    category : str
        Categorical variable used for grouping.

    min_samples : int, default=0
        Minimum number of samples required for a category to be retained.

    Returns
    -------
    pandas.DataFrame
        Category-level sample count, adverse count and adverse rate.
    """
    risk = (
        df.groupby(category)["Target_Adverse"]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )

    risk["Adverse_Rate"] = risk["mean"] * 100

    if min_samples > 0:
        risk = risk[
            risk["count"] >= min_samples
        ].copy()

    return risk


def plot_adverse_rate(
    df,
    category,
    title,
    ylabel=None,
    min_samples=0,
    top_n=None,
):
    """
    Plot observed adverse rate for a categorical variable.

    This reusable function represents the same approach used in the
    notebook for product group, specific product, country of origin,
    business type, sampling programme and processing characteristic.
    """
    risk = calculate_adverse_rate(
        df,
        category,
        min_samples=min_samples,
    )

    if top_n is not None:
        risk = (
            risk
            .sort_values(
                "Adverse_Rate",
                ascending=False,
            )
            .head(top_n)
        )

    risk = risk.sort_values(
        "Adverse_Rate",
        ascending=True,
    )

    plt.figure(figsize=(11, 8))

    ax = sns.barplot(
        data=risk,
        x="Adverse_Rate",
        y=category,
    )

    plt.title(title)
    plt.xlabel("Adverse Samples (%)")
    plt.ylabel(ylabel or category)

    for i, value in enumerate(
        risk["Adverse_Rate"]
    ):
        ax.text(
            value,
            i,
            f" {value:.2f}%",
            va="center",
        )

    plt.tight_layout()
    return ax


def plot_monthly_adverse_rate(model_df):
    """
    Plot observed adverse rate by sampling month.
    """
    monthly_risk = (
        model_df
        .groupby("Sampling_Month")["Target_Adverse"]
        .agg(["count", "sum", "mean"])
        .reindex(range(1, 13))
    )

    monthly_risk["Adverse_Rate"] = (
        monthly_risk["mean"] * 100
    )

    month_names = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec",
    ]

    plt.figure(figsize=(10, 5))

    ax = sns.barplot(
        x=month_names,
        y=monthly_risk["Adverse_Rate"].values,
    )

    plt.title(
        "Observed Adverse Rate by Sampling Month"
    )
    plt.xlabel("Sampling Month")
    plt.ylabel("Adverse Samples (%)")

    for i, value in enumerate(
        monthly_risk["Adverse_Rate"].values
    ):
        ax.text(
            i,
            value + 0.03,
            f"{value:.2f}%",
            ha="center",
        )

    plt.tight_layout()
    return ax
