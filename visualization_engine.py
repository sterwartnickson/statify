import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# STATIFY VISUALIZATION ENGINE
# ============================================================


def create_histogram(data, variable_name):
    """
    Create a histogram for a numerical variable.
    """

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        data,
        bins="auto",
        density=True,
        alpha=0.7
    )

    ax.set_title(
        f"Distribution of {variable_name}"
    )

    ax.set_xlabel(variable_name)
    ax.set_ylabel("Density")

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    return fig


def create_boxplot(data, variable_name):
    """
    Create a boxplot for a numerical variable.
    """

    fig, ax = plt.subplots(figsize=(9, 3))

    ax.boxplot(
        data,
        vert=False
    )

    ax.set_title(
        f"Boxplot of {variable_name}"
    )

    ax.set_xlabel(variable_name)

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    return fig


def _unavailable_plot(title, message):
    """Return a readable placeholder figure instead of raising."""
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.5, 0.5, message,
        ha="center", va="center", wrap=True, fontsize=11
    )
    fig.tight_layout()
    return fig


def create_density_plot(data, variable_name):
    """
    Create a kernel density estimate plot.

    A KDE needs at least two distinct values, so constant or nearly empty
    variables get a short explanatory figure rather than an exception.
    """

    title = f"Density Plot of {variable_name}"

    values = np.asarray(data, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2 or np.unique(values).size < 2:
        return _unavailable_plot(
            title,
            "A density plot needs at least two different values."
        )

    try:
        density = stats.gaussian_kde(values)
    except (np.linalg.LinAlgError, ValueError):
        return _unavailable_plot(
            title,
            "A density estimate could not be computed for this variable."
        )

    fig, ax = plt.subplots(figsize=(9, 5))

    x_values = np.linspace(
        np.min(values),
        np.max(values),
        500
    )

    y_values = density(x_values)

    ax.plot(
        x_values,
        y_values
    )

    ax.fill_between(
        x_values,
        y_values,
        alpha=0.25
    )

    ax.set_title(title)

    ax.set_xlabel(variable_name)
    ax.set_ylabel("Density")

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    return fig


def create_qq_plot(data, variable_name):
    """
    Create a normal Q-Q plot.
    """

    fig, ax = plt.subplots(figsize=(7, 7))

    stats.probplot(
        data,
        dist="norm",
        plot=ax
    )

    ax.set_title(
        f"Normal Q-Q Plot: {variable_name}"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    return fig


def detect_iqr_outliers(data):
    """
    Detect potential outliers using the IQR rule.
    """

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]

    if data.size == 0:
        return {
            "Q1": np.nan,
            "Q3": np.nan,
            "IQR": np.nan,
            "Lower Bound": np.nan,
            "Upper Bound": np.nan,
            "Outlier Count": 0,
            "Outliers": data
        }

    q1 = np.percentile(
        data,
        25
    )

    q3 = np.percentile(
        data,
        75
    )

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = data[
        (data < lower_bound) |
        (data > upper_bound)
    ]

    return {
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": len(outliers),
        "Outliers": outliers
    }
