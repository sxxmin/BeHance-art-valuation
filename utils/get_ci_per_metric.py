import argparse
import os

import numpy as np
import pandas as pds
import joblib
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

from tqdm import tqdm

# =====================================================================
# [Uncertainty and evaluation metrics]
#   - Load already-trained outputs ({'y','y_hat'} in *.values)
#   - Bootstrap the TEST set (resample with replacement, size = n_test)
#   - Report point estimate + bootstrap mean/median + [2.5%, 97.5%] CI:
#       R^2, log-scale MAE, Pearson r, Spearman rs
#   - NO retraining. Mirrors Trainer.py path / naming conventions.
#
#   Point estimate = full-sample single computation (the value to report).
#   Bootstrap mean/median are stored ALONGSIDE it for reference; CI is the
#   2.5/97.5 percentile of the bootstrap distribution.
#
# NOTE on scale (matches Trainer.py):
#   For target in {appreciation, view}, stored y / y_hat are ALREADY
#   base-10 log-transformed (NBR predictions were also log10'd before
#   saving). Hence mean(|y - y_hat|) here IS the log-scale MAE, and
#   R^2 is computed on the log scale. No further transform is applied.
# =====================================================================

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='LR')            # {LR, NB, XGB, EBM}
parser.add_argument('--target', type=str, default='appreciation') # {appreciation, view}
parser.add_argument('--engaging_group', type=str, default='All')  # {Artist, Artwork, All}
parser.add_argument('--using_c_variable', type=str, default='False')  # matched as string to the saved filename
parser.add_argument('--window_opt', type=str, default='')

parser.add_argument('--n_boot', type=int, default=1000)
parser.add_argument('--seed', type=int, default=260812)  # shared seed -> paired resampling across cells
parser.add_argument('--ci_low', type=float, default=2.5)
parser.add_argument('--ci_high', type=float, default=97.5)

# run every Figure-1 cell in one pass (overrides the single-cell args above)
parser.add_argument('--run_all', action='store_true')

args = parser.parse_args()

record_date = '260821'
window_opt = args.window_opt

MODELS_DIR = f'../dataset/models/260812'
OUT_DIR = f'./dataset/models/{window_opt}/{record_date}_metrics_with_ci'
os.makedirs(OUT_DIR, exist_ok=True)

def _load_values(save_filename):
    """Load stored {'y', 'y_hat'} produced by Trainer.py."""
    path = f'{MODELS_DIR}/{save_filename}.values'
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    store = joblib.load(path)
    y = np.asarray(store['y'], dtype=float).ravel()
    y_hat = np.asarray(store['y_hat'], dtype=float).ravel()
    # guard against non-finite entries (e.g. any NBR log issues)
    mask = np.isfinite(y) & np.isfinite(y_hat)
    return y[mask], y_hat[mask]


def _point_metrics(y, y_hat):
    """R^2 uses sklearn convention (SS_tot around the TEST mean);
    a negative value therefore means the model does worse than
    predicting the test mean -- this is exactly how the negative
    NBR-artwork R^2 values should be read in the manuscript."""
    r2 = r2_score(y, y_hat)
    log_mae = mean_absolute_error(y, y_hat)  # y,y_hat already log10 -> log-scale MAE
    pear = pearsonr(y_hat, y)[0]
    spear = spearmanr(y_hat, y)[0]
    return r2, log_mae, pear, spear


def _bootstrap(y, y_hat, n_boot, seed, ci_low, ci_high):
    n = len(y)
    rng = np.random.default_rng(seed)  # shared seed keeps resampling paired across cells

    r2_b = np.empty(n_boot)
    mae_b = np.empty(n_boot)
    pear_b = np.empty(n_boot)
    spear_b = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)  # resample WITH replacement, size = n_test
        yb, yhb = y[idx], y_hat[idx]
        r2_b[b] = r2_score(yb, yhb)
        mae_b[b] = mean_absolute_error(yb, yhb)
        # correlations undefined if a resample is constant; guard with try
        try:
            pear_b[b] = pearsonr(yhb, yb)[0]
        except Exception:
            pear_b[b] = np.nan
        try:
            spear_b[b] = spearmanr(yhb, yb)[0]
        except Exception:
            spear_b[b] = np.nan

    def ci(arr):
        arr = arr[np.isfinite(arr)]
        return (np.percentile(arr, ci_low), np.percentile(arr, ci_high))

    def mean_(arr):
        return float(np.nanmean(arr))

    def median_(arr):
        return float(np.nanmedian(arr))

    return {
        'R2_ci': ci(r2_b),
        'logMAE_ci': ci(mae_b),
        'Pearson_ci': ci(pear_b),
        'Spearman_ci': ci(spear_b),
        # bootstrap distribution mean / median (kept alongside the point estimate)
        'R2_bmean': mean_(r2_b),
        'logMAE_bmean': mean_(mae_b),
        'Pearson_bmean': mean_(pear_b),
        'Spearman_bmean': mean_(spear_b),
        '_dist': {'R2': r2_b, 'logMAE': mae_b, 'Pearson': pear_b, 'Spearman': spear_b},
    }


def evaluate_one(model, target, engaging_group, using_c_variable):
    save_filename = f'{model}+{target}+{engaging_group}+{using_c_variable}'
    y, y_hat = _load_values(save_filename)

    r2, log_mae, pear, spear = _point_metrics(y, y_hat)
    boot = _bootstrap(y, y_hat, args.n_boot, args.seed, args.ci_low, args.ci_high)

    row = {
        'model': model,
        'target': target,
        'engaging_group': engaging_group,
        'n_test': len(y),
        # R2: point estimate (full-sample), bootstrap mean/median, CI
        'R2_bmean': boot['R2_bmean'],
        'R2_lo': boot['R2_ci'][0], 'R2_hi': boot['R2_ci'][1],
        # logMAE
        'logMAE_bmean': boot['logMAE_bmean'],
        'logMAE_lo': boot['logMAE_ci'][0], 'logMAE_hi': boot['logMAE_ci'][1],
        # Pearson
        'Pearson_bmean': boot['Pearson_bmean'],
        'Pearson_lo': boot['Pearson_ci'][0], 'Pearson_hi': boot['Pearson_ci'][1],
        # Spearman
        'Spearman_bmean': boot['Spearman_bmean'],
        'Spearman_lo': boot['Spearman_ci'][0], 'Spearman_hi': boot['Spearman_ci'][1]
    }
    return row, boot['_dist']


def main():
    if args.run_all:
        models = ['LR', 'NB', 'XGB', 'EBM']
        targets = ['appreciation', 'view']
        groups = ['Artist', 'Artwork', 'All']
        using_c = args.using_c_variable  # keep the same control-variable setting as the saved files

        rows = []
        dists = {}
        combos = [(m, t, g) for m in models for t in targets for g in groups]
        for (m, t, g) in tqdm(combos, desc='bootstrapping Figure 1 cells'):
            save_filename = f'{m}+{t}+{g}+{using_c}'
            try:
                row, dist = evaluate_one(m, t, g, using_c)
            except FileNotFoundError as e:
                print(f'[skip] missing: {e}')
                continue
            rows.append(row)
            dists[save_filename] = dist

        df = pds.DataFrame(rows)
        out_xlsx = f'{OUT_DIR}/figure1_bootstrap_metrics.xlsx'
        out_csv = f'{OUT_DIR}/figure1_bootstrap_metrics.csv'
        df.to_excel(out_xlsx, index=False)
        df.to_csv(out_csv, index=False)
        joblib.dump(dists, f'{OUT_DIR}/figure1_bootstrap_distributions.pkl')

        print(df.to_string(index=False))
        print(f'\nSaved: {out_xlsx}')
        print(f'Saved: {out_csv}')
        print(f'Saved bootstrap distributions: {OUT_DIR}/figure1_bootstrap_distributions.pkl')
    else:
        row, dist = evaluate_one(
            args.model, args.target, args.engaging_group, args.using_c_variable
        )
        df = pds.DataFrame([row])
        print(df.to_string(index=False))
        save_filename = f'{args.model}+{args.target}+{args.engaging_group}+{args.using_c_variable}'
        df.to_excel(f'{OUT_DIR}/{save_filename}.metrics.xlsx', index=False)
        joblib.dump(dist, f'{OUT_DIR}/{save_filename}.dist.pkl')
        print(f'\nSaved: {OUT_DIR}/{save_filename}.metrics.xlsx')


if __name__ == '__main__':
    main()