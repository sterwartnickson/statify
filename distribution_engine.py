import numpy as np
import pandas as pd

from scipy import stats


# ============================================================
# STATIFY DISTRIBUTION ENGINE
# ============================================================


SUPPORTED_DISTRIBUTIONS = {
    "Normal": stats.norm,
    "Lognormal": stats.lognorm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Uniform": stats.uniform,
}


def prepare_numeric_data(series):
    """
    Convert a variable to numeric data and remove missing values.
    """

    data = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    return data.to_numpy()


def distribution_shape(data):
    """
    Describe the general shape of a numerical distribution.
    """

    if len(data) < 3:
        return "Insufficient data"

    skewness = stats.skew(data)

    if abs(skewness) < 0.5:
        return "Approximately symmetric"

    elif skewness >= 0.5:
        return "Right-skewed"

    else:
        return "Left-skewed"


def fit_distributions(data):
    """
    Fit supported probability distributions and calculate
    goodness-of-fit metrics.

    AIC and BIC are calculated using the fitted log-likelihood.
    """

    results = []

    n = len(data)

    if n < 10:
        return pd.DataFrame()

    for name, distribution in SUPPORTED_DISTRIBUTIONS.items():

        try:

            # Some distributions require positive observations.
            if name in [
                "Lognormal",
                "Exponential",
                "Gamma",
                "Weibull"
            ] and np.any(data <= 0):

                continue

            params = distribution.fit(data)

            log_likelihood = np.sum(
                distribution.logpdf(
                    data,
                    *params
                )
            )

            number_of_parameters = len(params)

            aic = (
                2 * number_of_parameters
                - 2 * log_likelihood
            )

            bic = (
                number_of_parameters * np.log(n)
                - 2 * log_likelihood
            )

            ks_statistic, ks_pvalue = stats.kstest(
    data,
    distribution.cdf,
    args=params
)

            results.append({
                "Distribution": name,
                "AIC": aic,
                "BIC": bic,
                "KS Statistic": ks_statistic,
                "KS p-value": ks_pvalue,
            })

        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="AIC"
    ).reset_index(drop=True)

    return results_df


def best_distribution(results):
    """
    Select the candidate distribution with the lowest AIC.
    """

    if results.empty:
        return None

    return results.iloc[0]["Distribution"]


def normality_tests(data):
    """
    Perform common normality tests.
    """

    results = {}

    if len(data) >= 3:

        try:

            shapiro_stat, shapiro_p = stats.shapiro(
                data
            )

            results["Shapiro-Wilk Statistic"] = shapiro_stat
            results["Shapiro-Wilk p-value"] = shapiro_p

        except Exception:
            pass

    if len(data) >= 20:

        try:

            dagostino_stat, dagostino_p = (
                stats.normaltest(data)
            )

            results["D'Agostino Statistic"] = dagostino_stat
            results["D'Agostino p-value"] = dagostino_p

        except Exception:
            pass

    return results


def interpret_normality(p_value, alpha=0.05):
    """
    Interpret a normality test cautiously.
    """

    if p_value is None or pd.isna(p_value):

        return "Normality could not be assessed."

    if p_value >= alpha:

        return (
            "There is insufficient evidence to reject "
            "normality at the selected significance level."
        )

    return (
        "There is evidence against normality at the "
        "selected significance level."
    )