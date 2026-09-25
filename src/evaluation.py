import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error


def evaluate_predictions(y_true, y_pred):
    """
    Calculate Spearman rank correlation and MSE in original DMS_score units.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    spearman_corr, _ = spearmanr(y_true, y_pred)
    mse_val = mean_squared_error(y_true, y_pred)

    return {
        "spearman": float(spearman_corr),
        "mse": float(mse_val),
    }


def format_results_table(results_dict, split_name="Validation"):
    """
    Format a clean markdown table summarizing results for multiple methods.
    """
    lines = [
        f"### {split_name} Results",
        "",
        "| Method | Spearman $\\rho$ | MSE |",
        "| :--- | :---: | :---: |",
    ]
    for method_name, metrics in results_dict.items():
        lines.append(f"| {method_name} | {metrics['spearman']:.4f} | {metrics['mse']:.4f} |")
    return "\n".join(lines)
