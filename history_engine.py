import pandas as pd
from datetime import datetime


# ============================================================
# STATIFY REPRODUCIBILITY ENGINE
# ============================================================


def create_history():

    return []


def add_history(
    history,
    action_info
):

    entry = {
        "Step": len(history) + 1,
        "Time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "Action": action_info.get(
            "action",
            "Unknown"
        ),
        "Details": action_info.get(
            "details",
            ""
        ),
        "Affected Rows": action_info.get(
            "affected_rows",
            0
        )
    }

    history.append(
        entry
    )

    return history


def history_table(history):

    if not history:

        return pd.DataFrame(
            columns=[
                "Step",
                "Time",
                "Action",
                "Details",
                "Affected Rows"
            ]
        )

    return pd.DataFrame(
        history
    )