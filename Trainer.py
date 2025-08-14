import argparse
import os

import numpy as np
import pandas as pds
import pickle as pkl
import statsmodels.api as sm
from str2bool import str2bool
import joblib
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='LR') # {LR, NB, XGB, EBM}
parser.add_argument('--target', type=str, default='view') # {appreciation, view}
parser.add_argument('--engaging_group', type=str, default='Artist') # {Artist, Artwork, All}
parser.add_argument('--using_c_variable', type=str2bool, default=True)
parser.add_argument('--window_opt', type=str, default='')

args = parser.parse_args()
record_date = '250603'
window_opt = args.window_opt

save_filename = f'{args.model}+{args.target}+{args.engaging_group}+{args.using_c_variable}'

# load total dataset
df = pds.concat([pds.read_csv(f'./dataset/variables_{i}.csv') for i in range(0, 15)]).reset_index(drop=True)


# get y information

ys = df[['appreciation', 'view']].copy()
del df['appreciation']; del df['view'];

  
if args.model != 'NB':
    y = np.log10(ys[args.target])
else:
    y = ys[args.target]
        
        
# get X information
a_cont_variable = 'Data_acquisition_to_project_publication_interval'
a_split_indicator = 'Artist_Project_publication_year_2020'

if args.engaging_group == 'All':
    X = df.copy()
elif args.engaging_group == 'Artist':
    X = df[
        [_col for _col in df.columns if args.engaging_group in _col]+[a_cont_variable]
    ].copy()
elif args.engaging_group == 'Artwork':
    X = df[
        [_col for _col in df.columns if args.engaging_group in _col]+[a_split_indicator, a_cont_variable]
    ].copy()
    

# split train & validation
train_indices = np.where(X[a_split_indicator] == 0)[0]
test_indices = np.where(X[a_split_indicator] == 1)[0]

X_trn, y_trn = X.iloc[train_indices], y[train_indices]
X_tst, y_tst = X.iloc[test_indices], y[test_indices]

if args.engaging_group == 'Artwork':
    del X_trn[a_split_indicator]; del X_tst[a_split_indicator]
    
if not args.using_c_variable:
    del X_trn[a_cont_variable]; del X_tst[a_cont_variable]
    
    
store_outputs = dict()

os.makedirs(f'./dataset/models/{window_opt}/{record_date}', exist_ok=True)
os.makedirs(f'./dataset/models/{window_opt}/{record_date}_results', exist_ok=True)
if args.model == 'LR':
    X_trn, X_tst = sm.add_constant(X_trn, prepend=False, has_constant='add'), sm.add_constant(X_tst, prepend=False, has_constant='add')
    model = sm.OLS(y_trn, X_trn)
    results = model.fit()
    store_outputs['y'], store_outputs['y_hat'] = y_tst.values, results.predict(X_tst).values

    results_summary = pds.DataFrame({
        'Parameter': results.params.index,
        'Coefficient': list(map(lambda x: f"{x:.4f}", results.params.values)),
        'Standard Error': list(map(lambda x: f"({x:.4f})", results.bse.values)),
        'P-value': list(map(lambda x: f"{x:.4f}", results.pvalues.values))
    })

    joblib.dump(X_trn.columns.values, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.column')
    joblib.dump(store_outputs, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.values')
    joblib.dump(results, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.models')
    results_summary.to_excel(f'./dataset/models/{window_opt}/{record_date}_results/{save_filename}.xlsx', index=False)
    
elif args.model == 'NB':
    X_trn, X_tst = sm.add_constant(X_trn, prepend=False, has_constant='add'), sm.add_constant(X_tst, prepend=False, has_constant='add')
    model = sm.GLM(y_trn, X_trn, family=sm.families.NegativeBinomial())
    results = model.fit()
    store_outputs['y'], store_outputs['y_hat'] = np.log10(y_tst.values), np.log10(results.predict(X_tst).values)

    results_summary = pds.DataFrame({
        'Parameter': results.params.index,
        'Coefficient': list(map(lambda x: f"{x:.4f}", results.params.values)),
        'Standard Error': list(map(lambda x: f"({x:.4f})", results.bse.values)),
        'P-value': list(map(lambda x: f"{x:.4f}", results.pvalues.values))
    })

    joblib.dump(X_trn.columns.values, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.column')
    joblib.dump(store_outputs, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.values')
    joblib.dump(results, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.models')
    results_summary.to_excel(f'./dataset/models/{window_opt}/{record_date}_results/{save_filename}.xlsx', index=False)
    
elif args.model == 'XGB':
    from xgboost.sklearn import XGBRegressor
    regressor = XGBRegressor(random_state=int(record_date))
    regressor.fit(X_trn, y_trn)
    y_hat = regressor.predict(X_tst)
    store_outputs['y'], store_outputs['y_hat'] = y_tst.values, y_hat

    joblib.dump(X_trn.columns.values, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.column')
    joblib.dump(store_outputs, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.values')
    joblib.dump(regressor, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.models')
    
elif args.model == 'EBM':
    from interpret.glassbox import ExplainableBoostingRegressor
    regressor = ExplainableBoostingRegressor(random_state=int(record_date))
    regressor.fit(X_trn, y_trn)
    y_hat = regressor.predict(X_tst)
    store_outputs['y'], store_outputs['y_hat'] = y_tst.values, y_hat

    joblib.dump(X_trn.columns.values, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.column')
    joblib.dump(store_outputs, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.values')
    joblib.dump(regressor, f'./dataset/models/{window_opt}/{record_date}/{save_filename}.models')
    
print(f'Complete: {save_filename}')
