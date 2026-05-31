import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
import warnings

# ── Config ───────────────────────────────────────────────────────────────────
TRAIN_PATH  = 'train.csv'
TEST_PATH   = 'test.csv'
OUTPUT_PATH = 'submission.csv'

D48_WEIGHT = 0.2
D49_WEIGHT = 10.0
LAG_OFFSETS = [15, -15, 30, -30, 60, -60, 120, -120]

LGB_PARAMS = dict(
    objective='regression', metric='rmse', num_leaves=255,
    learning_rate=0.02, feature_fraction=0.7, bagging_fraction=0.8,
    bagging_freq=5, min_child_samples=5, n_estimators=6000,
    verbosity=-1, random_state=42
)
XGB_PARAMS = dict(
    objective='reg:squarederror', max_depth=8, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, tree_method='hist',
    seed=42, verbosity=0
)

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# ── Load & parse ──────────────────────────────────────────────────────────────
train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)
for df in [train, test]:
    df['ts_min'] = df['timestamp'].apply(parse_ts)
    df['hour']   = df['ts_min'] // 60

d48 = train[train['day'] == 48].copy()
d49 = train[train['day'] == 49].copy()
print(f"D48: {len(d48)} rows  D49: {len(d49)} rows  Test: {len(test)} rows")

# ── Spatial coordinates ───────────────────────────────────────────────────────
all_geo = pd.concat([train['geohash'], test['geohash']]).unique()
cdf = pd.DataFrame({'geohash': all_geo})
cdf[['lat', 'lon']] = cdf['geohash'].apply(lambda g: pd.Series(decode_geohash(g)))

# ── All aggregations from D48 only ────────────────────────────────────────────
d48['g4'] = d48['geohash'].str[:4]
d48['g3'] = d48['geohash'].str[:3]

geo_ts = d48.groupby(['geohash', 'ts_min'])['demand'] \
             .agg(['sum', 'count', 'mean']).reset_index()
geo_ts.columns = ['geohash', 'ts_min', 'ts_sum', 'ts_count', 'geo_ts_mean']

geo_h = d48.groupby(['geohash', 'hour'])['demand'].mean().reset_index()
geo_h.columns = ['geohash', 'hour', 'geo_h_mean']

geo_s = d48.groupby('geohash')['demand'].agg(
    geo_mean='mean', geo_std='std', geo_max='max', geo_min='min'
).reset_index().fillna({'geo_std': 0})

hr_s = d48.groupby('hour')['demand'].agg(['mean', 'std']).reset_index()
hr_s.columns = ['hour', 'hr_mean', 'hr_std']

ts_s = d48.groupby('ts_min')['demand'].mean().reset_index()
ts_s.columns = ['ts_min', 'ts_mean']

rd_ts = d48.groupby(['RoadType', 'ts_min'])['demand'].mean().reset_index()
rd_ts.columns = ['RoadType', 'ts_min', 'rd_ts_mean']

rd_h = d48.groupby(['RoadType', 'hour'])['demand'].mean().reset_index()
rd_h.columns = ['RoadType', 'hour', 'rd_h_mean']

rd_s = d48.groupby('RoadType')['demand'].mean().reset_index()
rd_s.columns = ['RoadType', 'rd_mean']

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

GM  = d48['demand'].mean()
MT  = d48['Temperature'].median()

# ── Feature builder ───────────────────────────────────────────────────────────
def featurize(df, loo_d48=False):
    df = df.copy()
    for c in ['RoadType', 'Weather']:
        df[c] = df[c].fillna('Unknown')
    df['re']     = df['RoadType'].map({'Residential': 0, 'Street': 1,
                                        'Highway': 2, 'Unknown': -1}).fillna(-1)
    df['we']     = df['Weather'].map({'Sunny': 0, 'Rainy': 1,
                                       'Foggy': 2, 'Snowy': 3, 'Unknown': -1}).fillna(-1)
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

    df = df.merge(cdf,  on='geohash',              how='left')
    df = df.merge(geo_s, on='geohash',             how='left')
    df = df.merge(geo_ts[['geohash', 'ts_min', 'geo_ts_mean']],
                  on=['geohash', 'ts_min'],         how='left')
    df = df.merge(geo_h, on=['geohash', 'hour'],   how='left')
    df = df.merge(hr_s,  on='hour',                how='left')
    df = df.merge(ts_s,  on='ts_min',              how='left')
    df = df.merge(rd_ts, on=['RoadType', 'ts_min'],how='left')
    df = df.merge(rd_h,  on=['RoadType', 'hour'],  how='left')
    df = df.merge(rd_s,  on='RoadType',            how='left')
    df = df.merge(wt_h,  on=['Weather', 'hour'],   how='left')
    df = df.merge(p4_ts, on=['g4', 'ts_min'],      how='left')
    df = df.merge(p4_h,  on=['g4', 'hour'],        how='left')
    df = df.merge(p4_m,  on='g4',                  how='left')
    df = df.merge(p3_h,  on=['g3', 'hour'],        how='left')
    df = df.merge(ln_h,  on=['NumberofLanes', 'hour'], how='left')

    # Time-lag features (D48 geo_ts_mean shifted by offset)
    for off in LAG_OFFSETS:
        tmp = geo_ts[['geohash', 'ts_min', 'geo_ts_mean']].copy()
        tmp['ts_min'] = tmp['ts_min'] - off
        tmp.columns   = ['geohash', 'ts_min', f'gl{off:+d}']
        df = df.merge(tmp, on=['geohash', 'ts_min'], how='left')

    # LOO correction on geo_ts_mean for D48 training rows
    if loo_d48 and 'demand' in df.columns:
        t2 = df[['geohash', 'ts_min', 'demand']].merge(
            geo_ts, on=['geohash', 'ts_min'], how='left')
        df['geo_ts_mean'] = (
            (t2['ts_sum'] - t2['demand']) /
            (t2['ts_count'] - 1).clip(lower=1)
        ).values

    # Derived ratio features
    df['gtr'] = df['geo_ts_mean'] / (df['ts_mean']  + 1e-8)
    df['ghr'] = df['geo_h_mean']  / (df['hr_mean']  + 1e-8)
    df['gdr'] = df['geo_mean']    / (GM + 1e-8)
    df['rvh'] = df['rd_h_mean']   / (df['hr_mean']  + 1e-8)
    df['geo_ts_x_rd'] = df['geo_ts_mean'] * df['re']
    df['geo_h_x_ts']  = df['geo_h_mean']  * df['ts_mean']

    return df

print("Building features...")
fd48 = featurize(d48,  loo_d48=True)
fd49 = featurize(d49,  loo_d48=False)
ft   = featurize(test, loo_d48=False)
print("Done")

LAG_COLS = [f'gl{o:+d}' for o in LAG_OFFSETS]
FEAT_COLS = [
    'ts_min', 'hour', 'tss', 'tsc', 'hs', 'hc', 'td',
    'lat', 'lon', 're', 'we', 'lv_enc', 'lm_enc', 'Temperature', 'NumberofLanes',
    'geo_mean', 'geo_std', 'geo_max', 'geo_min',
    'geo_ts_mean', 'geo_h_mean',
    'hr_mean', 'hr_std', 'ts_mean',
    'rd_ts_mean', 'rd_h_mean', 'rd_mean',
    'wt_h_mean', 'p4_ts_mean', 'p4_h_mean', 'p4_mean', 'p3_h_mean', 'ln_h_mean',
    'gtr', 'ghr', 'gdr', 'rvh', 'geo_ts_x_rd', 'geo_h_x_ts',
] + LAG_COLS

# Fill NaNs using D48 median as reference
gf = fd48[FEAT_COLS].median()
for df in [fd48, fd49, ft]:
    for c in FEAT_COLS:
        df[c] = df[c].fillna(gf[c])

X48 = fd48[FEAT_COLS].values;  y48 = d48['demand'].values
X49 = fd49[FEAT_COLS].values;  y49 = d49['demand'].values
Xt  = ft[FEAT_COLS].values

# ── Temporal validation split on D49 ─────────────────────────────────────────
ts_u   = sorted(d49['ts_min'].unique())
val_ts = ts_u[-3:]                         
vm     = d49['ts_min'].isin(val_ts)
X49tr, X49vl = X49[~vm.values], X49[vm.values]
y49tr, y49vl = y49[~vm.values], y49[vm.values]

# Training: D48 (low weight) + D49 early (high weight)
Xtr = np.vstack([X48, X49tr])
ytr = np.concatenate([y48, y49tr])
wtr = np.concatenate([
    np.ones(len(y48))   * D48_WEIGHT,
    np.ones(len(y49tr)) * D49_WEIGHT,
])

# Final train arrays (all data, for test prediction)
Xall = np.vstack([X48, X49])
yall = np.concatenate([y48, y49])
wall = np.concatenate([
    np.ones(len(y48)) * D48_WEIGHT,
    np.ones(len(y49)) * D49_WEIGHT,
])

print(f"\nTrain: {len(ytr)}  Val: {len(y49vl)}  Features: {len(FEAT_COLS)}")

# ── LightGBM ──────────────────────────────────────────────────────────────────
print("\n--- LightGBM ---")
m_lgb = lgb.LGBMRegressor(**LGB_PARAMS)
m_lgb.fit(
    Xtr, ytr, sample_weight=wtr,
    eval_set=[(X49vl, y49vl)],
    callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(500)]
)
vl_lgb = m_lgb.predict(X49vl)
sc_lgb = max(0, 100 * r2_score(y49vl, vl_lgb))
print(f"Val R²×100 = {sc_lgb:.4f}   best_iter = {m_lgb.best_iteration_}")

mf_lgb = lgb.LGBMRegressor(**{**LGB_PARAMS, 'n_estimators': m_lgb.best_iteration_ + 100})
mf_lgb.fit(Xall, yall, sample_weight=wall, callbacks=[lgb.log_evaluation(-1)])
p_lgb = mf_lgb.predict(Xt)

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("\n--- XGBoost ---")
dtr = xgb.DMatrix(Xtr, label=ytr, weight=wtr)
dvl = xgb.DMatrix(X49vl, label=y49vl)
dts = xgb.DMatrix(Xt)

m_xgb = xgb.train(
    XGB_PARAMS, dtr, 5000,
    evals=[(dvl, 'val')],
    early_stopping_rounds=200,
    verbose_eval=False
)
vl_xgb = m_xgb.predict(dvl)
sc_xgb = max(0, 100 * r2_score(y49vl, vl_xgb))
print(f"Val R²×100 = {sc_xgb:.4f}   best_iter = {m_xgb.best_iteration}")

dall = xgb.DMatrix(Xall, label=yall, weight=wall)
mf_xgb = xgb.train(XGB_PARAMS, dall, m_xgb.best_iteration + 100, verbose_eval=False)
p_xgb = mf_xgb.predict(dts)

# ── Ensemble weighted by val score ────────────────────────────────────────────
sc = np.array([sc_lgb, sc_xgb])
w  = sc / sc.sum()
print(f"\nEnsemble weights: LGB={w[0]:.3f}  XGB={w[1]:.3f}")

blend_val = w[0] * vl_lgb + w[1] * vl_xgb
print(f"Blend val R²×100 = {max(0, 100 * r2_score(y49vl, blend_val)):.4f}")

final = np.clip(w[0] * p_lgb + w[1] * p_xgb, 0, 1)
print(f"Test pred: mean={final.mean():.4f}  std={final.std():.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
sub = pd.DataFrame({'Index': test['Index'], 'demand': final})
sub.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved → {OUTPUT_PATH}  shape={sub.shape}")