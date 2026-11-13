from pathlib import Path

import pandas as pd


def save_summary(subject, cv_scores, test_scores, output_dir, extra_scores=None):
    rows = []
    for metric, values in cv_scores.items():
        rows.append(
            {
                "dataset": subject,
                "evaluation": "5-fold CV",
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(),
            }
        )
    for metric, value in test_scores.items():
        rows.append(
            {
                "dataset": subject,
                "evaluation": "test",
                "metric": metric,
                "mean": value,
                "std": 0.0,
            }
        )
    for evaluation, metrics in (extra_scores or {}).items():
        for metric, value in metrics.items():
            rows.append(
                {
                    "dataset": subject,
                    "evaluation": evaluation,
                    "metric": metric,
                    "mean": value,
                    "std": 0.0,
                }
            )
    path = Path(output_dir) / f"{subject}_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_sample_results(subject, X_test, y_test, prediction, probabilities, output_dir):
    results = X_test.copy()
    results.insert(0, "sample_id", results.index)
    results["actual"] = y_test.to_numpy()
    results["predicted"] = prediction
    results["correct"] = results["actual"] == results["predicted"]
    results["probability_pass"] = probabilities
    path = Path(output_dir) / f"{subject}_samples.csv"
    results.to_csv(path, index=False)
    return path
