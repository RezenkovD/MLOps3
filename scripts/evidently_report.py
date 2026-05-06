from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

ROOT = Path(__file__).resolve().parent.parent


def main():
    ref_data = joblib.load(ROOT / "reference_stats.joblib")
    ref_df = pd.DataFrame(ref_data["X"], columns=ref_data["feature_names"])

    rng = np.random.default_rng(0)
    current = ref_df.copy().sample(n=200, random_state=0, replace=True)
    current["petal_length"] = current["petal_length"] + 1.5

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=current)
    out = ROOT / "drift_report.html"
    report.save_html(str(out))
    print(f"Report saved to {out}")


if __name__ == "__main__":
    main()
