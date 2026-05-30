"""
Traffic Demand Forecasting — Solution v6
=========================================
Key design decisions:
  1. All aggregation features come from Day 48 (the historical day).
  2. Training set = D48 (weight 0.3) + D49 hours 0-2 (weight 8).
     D48 covers ALL hour/feature ranges so the model doesn't extrapolate OOD;
     D49 calibrates the demand *level* for the prediction day.
  3. Validation proxy = D48 hours 2-13 (same hour range as test, avoiding the
     D49-hours-0-2 / test-hours-2-13 distribution shift that fooled earlier runs).
  4. LOO correction on geo_ts_mean for D48 training rows to prevent target leakage.
  5. Ensemble of LGB / XGB / CatBoost weighted by proxy-val R².

Proxy val scores: LGB 94.54 | XGB 92.14 | CAT 88.37
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────
TRAIN_PATH = 'train.csv'
TEST_PATH  = 'test.csv'
OUTPUT_PATH = 'submission.csv'

D48_WEIGHT = 0.3
D49_WEIGHT = 8.0

LGB_PARAMS = dict(objective='regression', metric='rmse', num_leaves=255,
                  learning_rate=0.03, feature_fraction=0.7, bagging_fraction=0.8,
                  bagging_freq=5, min_child_samples=5, n_estimators=5000,
                  verbosity=-1, random_state=42, lambda_l1=0.05, lambda_l2=0.1)

XGB_PARAMS = dict(objective='reg:squarederror', max_depth=8, learning_rate=0.03,
                  subsample=0.8, colsample_bytree=0.7, tree_method='hist',
                  seed=42, verbosity=0, reg_lambda=1, reg_alpha=0.05)

CAT_PARAMS = dict(iterations=3000, learning_rate=0.03, depth=8,
                  loss_function='RMSE', random_seed=42, verbose=0,
                  l2_leaf_reg=3, min_data_in_leaf=5)

# ── Helpers ────────────────────────────────────────────────────────────────
def parse_ts(ts):
    h, m = ts.split(':')
    return int(h) * 60 + int(m)

def decode_geohash(gh):
    M = {c: i for i, c in enumerate('0123456789bcdefghjkmnpqrstuvwxyz')}
    la, lo = [-90., 90.], [-180., 180.]
    il = True
    for c in gh:
        b = M.get(c, 0)
        for i in range(4, -1, -1):
            bit = (b >> i) & 1
            if il:
                mid = (lo[0] + lo[1]) / 2
                lo[0 if bit else 1] = mid
            else:
                mid = (la[0] + la[1]) / 2
                la[0 if bit else 1] = mid
            il = not il
    return (la[0] + la[1]) / 2, (lo[0] + lo[1]) / 2

# ── Load data ──────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

for df in [train, test]:
    df['ts_min'] = df['timestamp'].apply(parse_ts)
    df['hour']   = df['ts_min'] // 60

d48 = train[train['day'] == 48].copy()
d49 = train[train['day'] == 49].copy()

print(f"D48: {len(d48)} rows  hours {d48['hour'].min()}-{d48['hour'].max()}")
print(f"D49: {len(d49)} rows  hours {d49['hour'].min()}-{d49['hour'].max()}")
print(f"Test:{len(test)} rows  hours {test['hour'].min()}-{test['hour'].max()}")

# ── Spatial coordinates ────────────────────────────────────────────────────
all_geo = pd.concat([train['geohash'], test['geohash']]).unique()
cdf = pd.DataFrame({'geohash': all_geo})
cdf[['lat', 'lon']] = cdf['geohash'].apply(lambda g: pd.Series(decode_geohash(g)))

# ── Aggregations (all from D48) ────────────────────────────────────────────
d48['g4'] = d48['geohash'].str[:4]
d48['g3'] = d48['geohash'].str[:3]

geo_ts = (d48.groupby(['geohash', 'ts_min'])['demand']
            .agg(['sum', 'count', 'mean']).reset_index())
geo_ts.columns = ['geohash', 'ts_min', 'ts_sum', 'ts_count', 'geo_ts_mean']

geo_h = d48.groupby(['geohash', 'hour'])['demand'].mean().reset_index()
geo_h.columns = ['geohash', 'hour', 'geo_h_mean']

geo_s = (d48.groupby('geohash')['demand']
           .agg(geo_mean='mean', geo_std='std', geo_max='max', geo_min='min',
                geo_p25=lambda x: x.quantile(.25),
                geo_p75=lambda x: x.quantile(.75))
           .reset_index())

hr_s = d48.groupby('hour')['demand'].agg(['mean', 'std', 'median']).reset_index()
hr_s.columns = ['hour', 'hr_mean', 'hr_std', 'hr_med']

ts_s = d48.groupby('ts_min')['demand'].mean().reset_index()
ts_s.columns = ['ts_min', 'ts_mean']

rd_ts = d48.groupby(['RoadType', 'ts_min'])['demand'].mean().reset_index()
rd_ts.columns = ['RoadType', 'ts_min', 'rd_ts_mean']

rd_h = d48.groupby(['RoadType', 'hour'])['demand'].mean().reset_index()
rd_h.columns = ['RoadType', 'hour', 'rd_h_mean']

rd_s = d48.groupby('RoadType')['demand'].agg(['mean', 'std']).reset_index()
rd_s.columns = ['RoadType', 'rd_mean', 'rd_std']

wt_h = d48.groupby(['Weather', 'hour'])['demand'].mean().reset_index()
wt_h.columns = ['Weather', 'hour', 'wt_h_mean']

p4_ts = d48.groupby(['g4', 'ts_min'])['demand'].mean().reset_index()
p4_ts.columns = ['g4', 'ts_min', 'p4_ts_mean']

p4_h = d48.groupby(['g4', 'hour'])['demand'].mean().reset_index()
p4_h.columns = ['g4', 'hour', 'p4_h_mean']

p4_m = d48.groupby('g4')['demand'].mean().reset_index()
p4_m.columns = ['g4', 'p4_mean']

p3_h = d48.groupby(['g3', 'hour'])['demand'].mean().reset_index()
p3_h.columns = ['g3', 'hour', 'p3_h_mean']

ln_h = d48.groupby(['NumberofLanes', 'hour'])['demand'].mean().reset_index()
ln_h.columns = ['NumberofLanes', 'hour', 'ln_h_mean']

lv_h = d48.groupby(['LargeVehicles', 'hour'])['demand'].mean().reset_index()
lv_h.columns = ['LargeVehicles', 'hour', 'lv_h_mean']

GM = d48['demand'].mean()
MT = d48['Temperature'].median()

# D49/D48 ratio feature (calibrates demand level for prediction day)
d49_h = d49.groupby('hour')['demand'].mean().reset_index(); d49_h.columns = ['hour', 'd49_h_mean']
d48_h = d48.groupby('hour')['demand'].mean().reset_index(); d48_h.columns = ['hour', 'd48_h_mean']
h_ratio = d49_h.merge(d48_h, on='hour', how='left')
h_ratio['d49_d48_ratio'] = h_ratio['d49_h_mean'] / (h_ratio['d48_h_mean'] + 1e-8)
global_ratio = d49['demand'].mean() / d48[d48['hour'].isin(d49['hour'].unique())]['demand'].mean()

# ── Feature builder ────────────────────────────────────────────────────────
LAG_OFFSETS = [15, -15, 30, -30, 60, -60, 120, -120, 180, -180]
LAG_COLS    = [f'gl{o:+d}' for o in LAG_OFFSETS]

FEAT_COLS = [
    'ts_min', 'hour', 'tss', 'tsc', 'hs', 'hc', 'td',
    'lat', 'lon', 're', 'we', 'lv_enc', 'lm_enc', 'Temperature', 'NumberofLanes',
    'geo_mean', 'geo_std', 'geo_max', 'geo_min', 'geo_p25', 'geo_p75',
    'geo_ts_mean', 'geo_h_mean',
    'hr_mean', 'hr_std', 'hr_med', 'ts_mean',
    'rd_ts_mean', 'rd_h_mean', 'rd_mean', 'rd_std',
    'wt_h_mean', 'p4_ts_mean', 'p4_h_mean', 'p4_mean', 'p3_h_mean',
    'ln_h_mean', 'lv_h_mean',
    'gtr', 'ghr', 'gdr', 'rvh', 'geo_ts_x_rd', 'geo_h_x_ts',
    'd49_d48_ratio',
] + LAG_COLS

def featurize(df, loo=False):
    df = df.copy()
    for c in ['RoadType', 'Weather']:
        df[c] = df[c].fillna('Unknown')
    df['re']     = df['RoadType'].map({'Residential': 0, 'Street': 1, 'Highway': 2, 'Unknown': -1}).fillna(-1)
    df['we']     = df['Weather'].map({'Sunny': 0, 'Rainy': 1, 'Foggy': 2, 'Snowy': 3, 'Unknown': -1}).fillna(-1)
    df['lv_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['lm_enc'] = (df['Landmarks'] == 'Yes').astype(int)
    df['Temperature'] = df['Temperature'].fillna(MT)
    df['td']  = df['Temperature'] - MT
    df['tss'] = np.sin(2 * np.pi * df['ts_min'] / (24 * 60))
    df['tsc'] = np.cos(2 * np.pi * df['ts_min'] / (24 * 60))
    df['hs']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hc']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['g4']  = df['geohash'].str[:4]
    df['g3']  = df['geohash'].str[:3]

    df = df.merge(cdf, on='geohash', how='left')
    df = df.merge(geo_s, on='geohash', how='left')
    df = df.merge(geo_ts[['geohash', 'ts_min', 'geo_ts_mean']], on=['geohash', 'ts_min'], how='left')
    df = df.merge(geo_h, on=['geohash', 'hour'], how='left')
    df = df.merge(hr_s, on='hour', how='left')
    df = df.merge(ts_s, on='ts_min', how='left')
    df = df.merge(rd_ts, on=['RoadType', 'ts_min'], how='left')
    df = df.merge(rd_h, on=['RoadType', 'hour'], how='left')
    df = df.merge(rd_s, on='RoadType', how='left')
    df = df.merge(wt_h, on=['Weather', 'hour'], how='left')
    df = df.merge(p4_ts, on=['g4', 'ts_min'], how='left')
    df = df.merge(p4_h, on=['g4', 'hour'], how='left')
    df = df.merge(p4_m, on='g4', how='left')
    df = df.merge(p3_h, on=['g3', 'hour'], how='left')
    df = df.merge(ln_h, on=['NumberofLanes', 'hour'], how='left')
    df = df.merge(lv_h, on=['LargeVehicles', 'hour'], how='left')

    for off in LAG_OFFSETS:
        tmp = geo_ts[['geohash', 'ts_min', 'geo_ts_mean']].copy()
        tmp['ts_min'] = tmp['ts_min'] - off
        tmp.columns = ['geohash', 'ts_min', f'gl{off:+d}']
        df = df.merge(tmp, on=['geohash', 'ts_min'], how='left')

    # LOO correction on geo_ts_mean for D48 rows (prevents target leakage)
    if loo and 'demand' in df.columns:
        t2 = df[['geohash', 'ts_min', 'demand']].merge(geo_ts, on=['geohash', 'ts_min'], how='left')
        df['geo_ts_mean'] = ((t2['ts_sum'] - t2['demand']) / (t2['ts_count'] - 1).clip(lower=1)).values

    df['gtr']         = df['geo_ts_mean'] / (df['ts_mean'] + 1e-8)
    df['ghr']         = df['geo_h_mean']  / (df['hr_mean']  + 1e-8)
    df['gdr']         = df['geo_mean']    / (GM + 1e-8)
    df['rvh']         = df['rd_h_mean']   / (df['hr_mean']  + 1e-8)
    df['geo_ts_x_rd'] = df['geo_ts_mean'] * df['re']
    df['geo_h_x_ts']  = df['geo_h_mean']  * df['ts_mean']

    df = df.merge(h_ratio[['hour', 'd49_d48_ratio']], on='hour', how='left')
    df['d49_d48_ratio'] = df['d49_d48_ratio'].fillna(global_ratio)
    return df

print("Building features...")
fd48 = featurize(d48, loo=True)
fd49 = featurize(d49, loo=False)
ft   = featurize(test, loo=False)
print("Done")

gf = fd48[FEAT_COLS].median()
for df in [fd48, fd49, ft]:
    for c in FEAT_COLS:
        df[c] = df[c].fillna(gf[c])

X48  = fd48[FEAT_COLS].values;  y48  = d48['demand'].values
X49  = fd49[FEAT_COLS].values;  y49  = d49['demand'].values
Xt   = ft[FEAT_COLS].values

# ── Validation split (proxy for test scenario) ─────────────────────────────
# Train on D48 h0-1 + D49 → val on D48 h2-13 (same hour range as test)
val_m = (d48['hour'] >= 2) & (d48['hour'] <= 13)
tr_m  = d48['hour'] <= 1

Xtr = np.vstack([X48[tr_m.values], X49])
ytr = np.concatenate([y48[tr_m.values], y49])
wtr = np.concatenate([np.ones(tr_m.sum()) * D48_WEIGHT,
                       np.ones(len(y49))  * D49_WEIGHT])
Xvl = X48[val_m.values]
yvl = y48[val_m.values]

# Full train arrays (for final refit)
Xall = np.vstack([X48, X49])
yall = np.concatenate([y48, y49])
wall = np.concatenate([np.ones(len(y48)) * D48_WEIGHT,
                        np.ones(len(y49)) * D49_WEIGHT])

print(f"\nProxy val: train={len(ytr)}  val={len(yvl)}")

# ── LightGBM ───────────────────────────────────────────────────────────────
print("\n--- LightGBM ---")
m_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
m_lgb.fit(Xtr, ytr, sample_weight=wtr, eval_set=[(Xvl, yvl)],
          callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(500)])
vl_lgb = m_lgb.predict(Xvl)
sc_lgb = max(0, 100 * r2_score(yvl, vl_lgb))
print(f"Proxy val R²×100 = {sc_lgb:.4f}  (best_iter={m_lgb.best_iteration_})")

mf_lgb = lgb.LGBMRegressor(**{**LGB_PARAMS, 'n_estimators': m_lgb.best_iteration_ + 100})
mf_lgb.fit(Xall, yall, sample_weight=wall, callbacks=[lgb.log_evaluation(-1)])
p_lgb = mf_lgb.predict(Xt)

# ── XGBoost ───────────────────────────────────────────────────────────────
print("\n--- XGBoost ---")
dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
dvl = xgb.DMatrix(Xvl, label=yvl)
dts = xgb.DMatrix(Xt)
m_xgb = xgb.train(XGB_PARAMS, dtr, 4000, [(dvl, 'val')],
                   early_stopping_rounds=200, verbose_eval=False)
vl_xgb = m_xgb.predict(dvl)
sc_xgb = max(0, 100 * r2_score(yvl, vl_xgb))
print(f"Proxy val R²×100 = {sc_xgb:.4f}  (best_iter={m_xgb.best_iteration})")

dall = xgb.DMatrix(Xall, label=yall, weight=wall)
mf_xgb = xgb.train(XGB_PARAMS, dall, m_xgb.best_iteration + 100, verbose_eval=False)
p_xgb = mf_xgb.predict(dts)

# ── CatBoost ──────────────────────────────────────────────────────────────
print("\n--- CatBoost ---")
m_cat = CatBoostRegressor(**CAT_PARAMS)
m_cat.fit(Xtr, ytr, sample_weight=wtr, eval_set=(Xvl, yvl), early_stopping_rounds=200)
vl_cat = m_cat.predict(Xvl)
sc_cat = max(0, 100 * r2_score(yvl, vl_cat))
print(f"Proxy val R²×100 = {sc_cat:.4f}  (best_iter={m_cat.best_iteration_})")

mf_cat = CatBoostRegressor(**{**CAT_PARAMS, 'iterations': m_cat.best_iteration_ + 100, 'verbose': 0})
mf_cat.fit(Xall, yall, sample_weight=wall)
p_cat = mf_cat.predict(Xt)

# ── Ensemble weighted by proxy-val score ───────────────────────────────────
scores = np.array([sc_lgb, sc_xgb, sc_cat])
w      = scores / scores.sum()
print(f"\nEnsemble weights: LGB={w[0]:.3f}  XGB={w[1]:.3f}  CAT={w[2]:.3f}")

final = np.clip(w[0] * p_lgb + w[1] * p_xgb + w[2] * p_cat, 0, 1)
blend_val = w[0] * vl_lgb + w[1] * vl_xgb + w[2] * vl_cat
print(f"Blend proxy val R²×100 = {max(0,100*r2_score(yvl, blend_val)):.4f}")
print(f"Test pred: mean={final.mean():.4f}  std={final.std():.4f}")

# ── Save submission ────────────────────────────────────────────────────────
sub = pd.DataFrame({'Index': test['Index'], 'demand': final})
sub.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH}  shape={sub.shape}")
