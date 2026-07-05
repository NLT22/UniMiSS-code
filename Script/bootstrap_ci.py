"""Bootstrap 95% CIs on pooled out-of-fold predictions (no new compute/training).

Resamples the pooled test predictions (with replacement) per method and
recomputes AUC/AP, since a plain fold mean +/- std understates uncertainty
at this sample size (see Vabalas et al. 2019).

Usage:
    python bootstrap_ci.py --predictions results/combined/combined_predictions.csv \
        --output results/combined/summary/bootstrap_ci.csv --n-boot 2000
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    with open(args.predictions, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    rng = np.random.default_rng(args.seed)
    out_rows = []
    for method, method_rows in sorted(by_method.items()):
        y_true = np.array([int(r["y_true_abnormal"]) for r in method_rows])
        y_score = np.array([float(r["y_score_abnormal"]) for r in method_rows])
        n = len(y_true)

        point_auc = roc_auc_score(y_true, y_score)
        point_ap = average_precision_score(y_true, y_score)

        boot_aucs, boot_aps = [], []
        for _ in range(args.n_boot):
            idx = rng.integers(0, n, n)
            yt, ys = y_true[idx], y_score[idx]
            if len(set(yt)) < 2:
                continue
            boot_aucs.append(roc_auc_score(yt, ys))
            boot_aps.append(average_precision_score(yt, ys))

        auc_lo, auc_hi = np.percentile(boot_aucs, [2.5, 97.5])
        ap_lo, ap_hi = np.percentile(boot_aps, [2.5, 97.5])

        out_rows.append({
            "method": method, "n_pooled_test": n,
            "auc": round(point_auc, 4), "auc_ci_lo": round(auc_lo, 4), "auc_ci_hi": round(auc_hi, 4),
            "ap": round(point_ap, 4), "ap_ci_lo": round(ap_lo, 4), "ap_ci_hi": round(ap_hi, 4),
        })
        print(f"{method}: AUC={point_auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]  "
              f"AP={point_ap:.3f} [{ap_lo:.3f}, {ap_hi:.3f}]")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
