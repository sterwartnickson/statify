import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan


# ============================================================
# STATIFY REGRESSION ENGINE
# Linear regression + Binary logistic regression
# ============================================================


MAX_CATEGORY_LEVELS = 15


# ============================================================
# DESIGN MATRIX
# ============================================================

def _build_design_matrix(data, predictors):
    """
    Numeric predictors are used as-is.
    Non-numeric predictors are dummy-coded (first level = reference).
    Returns (X without constant, notes, error).
    """

    columns = []
    notes = []

    for name in predictors:

        series = data[name]

        if series.nunique() < 2:
            return None, notes, (
                f"Predictor '{name}' has only one distinct value "
                f"after removing missing rows."
            )

        is_numeric = (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        )

        if is_numeric:
            columns.append(series.astype(float).rename(name))
            continue

        series = series.astype(str)
        levels = sorted(series.unique())

        if len(levels) > MAX_CATEGORY_LEVELS:
            return None, notes, (
                f"Predictor '{name}' has {len(levels)} categories "
                f"(limit is {MAX_CATEGORY_LEVELS}). Group rare "
                f"categories first or choose a different predictor."
            )

        reference = levels[0]

        for level in levels[1:]:
            label = f"{name} = {level} (vs {reference})"
            columns.append(
                (series == level).astype(float).rename(label)
            )

        notes.append(
            f"'{name}' is categorical; reference level is '{reference}'."
        )

    X = pd.concat(columns, axis=1)

    return X, notes, None


def _vif_table(X):

    if X.shape[1] < 2:
        return None

    exog = sm.add_constant(X, has_constant="add")

    rows = []

    for i, name in enumerate(exog.columns):

        if name == "const":
            continue

        try:
            value = variance_inflation_factor(exog.values, i)
        except Exception:
            value = np.nan

        rows.append({"Predictor": name, "VIF": value})

    return pd.DataFrame(rows)


def _clean(data, outcome, predictors):

    columns = [outcome] + list(predictors)

    return data[columns].dropna().copy()


# ============================================================
# LINEAR REGRESSION
# ============================================================

def linear_regression(data, outcome, predictors):

    if not predictors:
        return {"error": "Select at least one predictor."}

    if outcome in predictors:
        return {"error": "The outcome cannot also be a predictor."}

    if not pd.api.types.is_numeric_dtype(data[outcome]):
        return {"error": "The outcome must be a numerical variable."}

    analysis_data = _clean(data, outcome, predictors)

    X, notes, error = _build_design_matrix(analysis_data, predictors)

    if error:
        return {"error": error}

    y = analysis_data[outcome].astype(float)

    if y.nunique() < 2:
        return {"error": "The outcome has no variation."}

    n = len(analysis_data)
    k = X.shape[1]

    if n <= k + 1:
        return {
            "error": (
                f"Not enough complete rows ({n}) for {k} model terms. "
                f"Use fewer predictors or more data."
            )
        }

    exog = sm.add_constant(X, has_constant="add")

    try:
        model = sm.OLS(y, exog).fit()
    except Exception as error:
        return {"error": str(error)}

    conf = model.conf_int()

    coefficients = pd.DataFrame({
        "Term": model.params.index,
        "Coefficient": model.params.values,
        "Std. Error": model.bse.values,
        "t": model.tvalues.values,
        "p-value": model.pvalues.values,
        "95% CI Lower": conf[0].values,
        "95% CI Upper": conf[1].values,
    })

    # ---- Assumption checks on residuals ----
    residuals = model.resid

    shapiro_p = np.nan

    if 3 <= len(residuals) <= 5000:
        try:
            shapiro_p = stats.shapiro(residuals).pvalue
        except Exception:
            pass

    try:
        bp_p = het_breuschpagan(residuals, exog)[1]
    except Exception:
        bp_p = np.nan

    diagnostics = pd.DataFrame({
        "Fitted": model.fittedvalues,
        "Residual": residuals,
    })

    return {
        "test": "Multiple linear regression",
        "model": model,
        "coefficients": coefficients,
        "n": n,
        "r_squared": model.rsquared,
        "adj_r_squared": model.rsquared_adj,
        "f_statistic": model.fvalue,
        "f_p_value": model.f_pvalue,
        "rmse": float(np.sqrt(model.mse_resid)),
        "shapiro_p": shapiro_p,
        "breusch_pagan_p": bp_p,
        "vif": _vif_table(X),
        "diagnostics": diagnostics,
        "notes": notes,
        "rows_dropped": len(data) - n,
    }


def interpret_linear_regression(result, alpha=0.05):

    lines = []

    if result["f_p_value"] < alpha:
        lines.append(
            f"The model as a whole is statistically significant "
            f"(F p = {result['f_p_value']:.4f}) and explains about "
            f"{result['r_squared'] * 100:.1f}% of the variation in the "
            f"outcome (adjusted R² = {result['adj_r_squared']:.3f})."
        )
    else:
        lines.append(
            f"The model as a whole is not statistically significant "
            f"(F p = {result['f_p_value']:.4f}); the predictors do not "
            f"explain the outcome better than chance."
        )

    table = result["coefficients"]
    table = table[table["Term"] != "const"]

    significant = table[table["p-value"] < alpha]

    if significant.empty:
        lines.append("No individual predictor is statistically significant.")
    else:
        for _, row in significant.iterrows():
            direction = "higher" if row["Coefficient"] > 0 else "lower"
            lines.append(
                f"• {row['Term']}: each one-unit increase is associated "
                f"with a {direction} outcome by "
                f"{abs(row['Coefficient']):.4g} (holding the other "
                f"predictors constant, p = {row['p-value']:.4f})."
            )

    return lines


def linear_regression_warnings(result, alpha=0.05):

    warnings = []

    if not pd.isna(result["shapiro_p"]) and result["shapiro_p"] < alpha:
        warnings.append(
            "Residuals deviate from normality (Shapiro-Wilk). "
            "Inference is usually robust with large samples, but consider "
            "a transformation or a robust method."
        )

    if not pd.isna(result["breusch_pagan_p"]) and \
            result["breusch_pagan_p"] < alpha:
        warnings.append(
            "Evidence of unequal residual variance "
            "(Breusch-Pagan). Standard errors may be unreliable."
        )

    vif = result["vif"]

    if vif is not None and (vif["VIF"] > 5).any():
        names = ", ".join(vif.loc[vif["VIF"] > 5, "Predictor"])
        warnings.append(
            f"High multicollinearity (VIF > 5): {names}. "
            f"Coefficients for these terms may be unstable."
        )

    if result["n"] < 10 * (len(result["coefficients"]) - 1):
        warnings.append(
            "The sample is small relative to the number of model terms "
            "(fewer than 10 rows per term). Results may be unstable."
        )

    return warnings


# ============================================================
# BINARY LOGISTIC REGRESSION
# ============================================================

def binary_outcome_candidates(data):
    """Columns that have exactly two distinct non-missing values."""

    return [
        column for column in data.columns
        if data[column].dropna().nunique() == 2
    ]


def _auc(y_true, scores):

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())

    if positives == 0 or negatives == 0:
        return np.nan

    ranks = stats.rankdata(scores)

    return (
        ranks[y_true == 1].sum()
        - positives * (positives + 1) / 2
    ) / (positives * negatives)


def binary_logistic_regression(
    data,
    outcome,
    predictors,
    positive_class=None
):

    if not predictors:
        return {"error": "Select at least one predictor."}

    if outcome in predictors:
        return {"error": "The outcome cannot also be a predictor."}

    analysis_data = _clean(data, outcome, predictors)

    levels = sorted(analysis_data[outcome].astype(str).unique())

    if len(levels) != 2:
        return {
            "error": (
                f"The outcome must have exactly two categories "
                f"(found {len(levels)})."
            )
        }

    if positive_class is None:
        positive_class = levels[-1]

    positive_class = str(positive_class)

    if positive_class not in levels:
        return {"error": "The selected event category is not in the data."}

    negative_class = [v for v in levels if v != positive_class][0]

    y = (
        analysis_data[outcome].astype(str) == positive_class
    ).astype(int)

    X, notes, error = _build_design_matrix(analysis_data, predictors)

    if error:
        return {"error": error}

    n = len(analysis_data)
    events = int(y.sum())
    non_events = n - events
    k = X.shape[1]

    if min(events, non_events) < 5:
        return {
            "error": (
                "Too few observations in one outcome category "
                f"({events} events, {non_events} non-events)."
            )
        }

    exog = sm.add_constant(X, has_constant="add")

    separation_message = (
        "The model could not be estimated reliably — a predictor "
        "perfectly (or almost perfectly) separates the outcome, or "
        "predictors are perfectly correlated. Remove or combine "
        "predictors."
    )

    try:
        with warnings.catch_warnings():
            # statsmodels only *warns* on separation / non-convergence,
            # so promote those warnings to errors and handle them.
            warnings.simplefilter("error")
            model = sm.Logit(y, exog).fit(disp=False, maxiter=200)

        if not model.mle_retvals.get("converged", True):
            return {"error": separation_message}

    except Exception as exc:
        name = type(exc).__name__

        if (
            "Separation" in name
            or "Convergence" in name
            or "Singular" in str(exc)
            or "LinAlg" in name
        ):
            return {"error": separation_message}

        return {"error": f"{name}: {exc}"}

    conf = model.conf_int()

    coefficients = pd.DataFrame({
        "Term": model.params.index,
        "Coefficient (log-odds)": model.params.values,
        "Std. Error": model.bse.values,
        "z": model.tvalues.values,
        "p-value": model.pvalues.values,
        "Odds Ratio": np.exp(np.clip(model.params.values, -700, 700)),
        "OR 95% CI Lower": np.exp(np.clip(conf[0].values, -700, 700)),
        "OR 95% CI Upper": np.exp(np.clip(conf[1].values, -700, 700)),
    })

    probabilities = model.predict(exog)
    predicted = (probabilities >= 0.5).astype(int)

    tp = int(((predicted == 1) & (y == 1)).sum())
    tn = int(((predicted == 0) & (y == 0)).sum())
    fp = int(((predicted == 1) & (y == 0)).sum())
    fn = int(((predicted == 0) & (y == 1)).sum())

    confusion = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=[f"Actual: {negative_class}", f"Actual: {positive_class}"],
        columns=[
            f"Predicted: {negative_class}",
            f"Predicted: {positive_class}",
        ],
    )

    return {
        "test": "Binary logistic regression",
        "model": model,
        "coefficients": coefficients,
        "n": n,
        "events": events,
        "non_events": non_events,
        "positive_class": positive_class,
        "negative_class": negative_class,
        "mcfadden_r2": model.prsquared,
        "llr_p_value": model.llr_pvalue,
        "aic": model.aic,
        "accuracy": (tp + tn) / n,
        "sensitivity": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "auc": _auc(y, probabilities),
        "confusion": confusion,
        "vif": _vif_table(X),
        "notes": notes,
        "rows_dropped": len(data) - n,
        "terms": k,
    }


def interpret_logistic_regression(result, alpha=0.05):

    lines = []

    if result["llr_p_value"] < alpha:
        lines.append(
            f"The model is statistically significant overall "
            f"(likelihood-ratio p = {result['llr_p_value']:.4f}). "
            f"It distinguishes '{result['positive_class']}' from "
            f"'{result['negative_class']}' with AUC = "
            f"{result['auc']:.3f}."
        )
    else:
        lines.append(
            f"The model is not statistically significant overall "
            f"(likelihood-ratio p = {result['llr_p_value']:.4f})."
        )

    table = result["coefficients"]
    table = table[table["Term"] != "const"]

    significant = table[table["p-value"] < alpha]

    if significant.empty:
        lines.append("No individual predictor is statistically significant.")
    else:
        for _, row in significant.iterrows():
            odds_ratio = row["Odds Ratio"]

            if odds_ratio >= 1:
                change = f"{(odds_ratio - 1) * 100:.1f}% higher"
            else:
                change = f"{(1 - odds_ratio) * 100:.1f}% lower"

            lines.append(
                f"• {row['Term']}: odds ratio = {odds_ratio:.3f} — "
                f"{change} odds of '{result['positive_class']}' per "
                f"one-unit increase (p = {row['p-value']:.4f})."
            )

    return lines


def logistic_regression_warnings(result):

    warnings = []

    epv = min(result["events"], result["non_events"]) / max(
        result["terms"], 1
    )

    if epv < 10:
        warnings.append(
            f"Only {epv:.1f} events per model term (10+ is recommended). "
            f"Estimates may be unstable or over-fitted."
        )

    vif = result["vif"]

    if vif is not None and (vif["VIF"] > 5).any():
        names = ", ".join(vif.loc[vif["VIF"] > 5, "Predictor"])
        warnings.append(
            f"High multicollinearity (VIF > 5): {names}."
        )

    warnings.append(
        "Accuracy and AUC are computed on the same data used to fit the "
        "model, so they are optimistic. Validate on new data for "
        "prediction work."
    )

    return warnings
