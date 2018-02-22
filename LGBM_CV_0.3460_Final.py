
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

df_train = pd.read_csv(
    'train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 36458909)  # 2016-01-01
)

df_test = pd.read_csv(
    "test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]  # , date_parser=parser
).set_index(
    ['store_nbr', 'item_nbr', 'date']
)

items = pd.read_csv(
    "items.csv",
).set_index("item_nbr")

df_2017 = df_train.loc[df_train.date>=pd.datetime(2015,8,1)]

promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train

df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
df_2017.columns = df_2017.columns.get_level_values(1)

items = items.reindex(df_2017.index.get_level_values(1))

def get_timespan(df, dt, minus, periods, freq='D'):
    d1 = pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)
    return df.iloc[:, df.columns.isin(d1)]

max_days = 140

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
        "unitssales_all_median": get_timespan_week(df_2017, t2017, max_days/7, max_days).median(axis=1).fillna(0).values,
        "unitssales_all_mean": get_timespan_week(df_2017, t2017, max_days/7, max_days).mean(axis=1).fillna(0).values,
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "1Q_7_2017": get_timespan(df_2017, t2017, 7, 7).quantile(0.25,axis=1).values,
        "med_7_2017": get_timespan(df_2017, t2017, 7, 7).median(axis=1).values,
        "3Q_7_2017": get_timespan(df_2017, t2017, 7, 7).quantile(0.75,axis=1).values,
        "min_7_2017": get_timespan(df_2017, t2017, 7, 7).min(axis=1).values,
        "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "1Q_14_2017": get_timespan(df_2017, t2017, 14, 14).quantile(0.25,axis=1).values,
        "med_14_2017": get_timespan(df_2017, t2017, 14, 14).median(axis=1).values,
        "3Q_14_2017": get_timespan(df_2017, t2017, 14, 14).quantile(0.75,axis=1).values,
        "1Q_21_2017": get_timespan(df_2017, t2017, 21, 21).quantile(0.25,axis=1).values,
        "mean_21_2017": get_timespan(df_2017, t2017, 21, 21).mean(axis=1).values,
        "med_21_2017": get_timespan(df_2017, t2017, 21, 21).median(axis=1).values,
        "3Q_21_2017": get_timespan(df_2017, t2017, 21, 21).quantile(0.75,axis=1).values,
        "1Q_28_2017": get_timespan(df_2017, t2017, 28, 28).quantile(0.25,axis=1).values,
        "mean_28_2017": get_timespan(df_2017, t2017, 28, 28).mean(axis=1).values,
        "3Q_28_2017": get_timespan(df_2017, t2017, 28, 28).quantile(0.75,axis=1).values,
        "mean_35_2017": get_timespan(df_2017, t2017, 35, 35).mean(axis=1).values,
        "max_35_2017": get_timespan(df_2017, t2017, 35, 35).max(axis=1).values,
        "mean_42_2017": get_timespan(df_2017, t2017, 42, 42).mean(axis=1).values,
        "mean_49_2017": get_timespan(df_2017, t2017, 49, 49).mean(axis=1).values,
        "1Q_70_2017": get_timespan(df_2017, t2017, 70, 70).quantile(0.25,axis=1).values,
        "max_70_2017": get_timespan(df_2017, t2017, 70, 70).max(axis=1).values,
        "3Q_70_2017": get_timespan(df_2017, t2017, 70, 70).quantile(0.75,axis=1).values,
        "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
        "mean_210_2017": get_timespan(df_2017, t2017, 210, 210).mean(axis=1).values,
        "promo_1_2017": get_timespan(promo_2017, t2017, 1, 1).sum(axis=1).values,
        "promo_3_2017": get_timespan(promo_2017, t2017, 3, 3).sum(axis=1).values,
        "promo_7_2017": get_timespan(promo_2017, t2017, 7, 7).sum(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values,
        "promo_210_2017": get_timespan(promo_2017, t2017, 210, 210).sum(axis=1).values
    })
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_7_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 49-i, 7, freq='7D').mean(axis=1).values
        X['mean_10_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').mean(axis=1).values
        X['mean_15_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 105-i, 15, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
        X['mean_30_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 210-i, 30, freq='7D').mean(axis=1).values
        X['mean_40_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 280-i, 40, freq='7D').mean(axis=1).values
        X['mean_50_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 350-i, 50, freq='7D').mean(axis=1).values


    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

print("Preparing dataset...")
t2017 = date(2017, 5, 17)
X_l, y_l = [], []
for i in range(8):
    print('step:',i+1)
#     if i>4:
#         t2017 = date(2016, 7, 13)
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = prepare_dataset(date(2017, 7, 26))


print("Training and predicting models...")
params = {
    'num_leaves': 2**8 - 1,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.025,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'num_threads': 6
}

MAX_ROUNDS = 5000
val_pred = []
test_pred = []
cate_vars = []
n_rounds = []
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * 8) * 0.25 + 1
    )
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=items["perishable"] * 0.25 + 1,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=50
    )
    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True
    )))
    val_pred.append(bst.predict(
        X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    n_rounds.append(bst.best_iteration)

# Val period : 2017, 7, 26 (data repeated 10 times)
# Unweighted validation mse:  0.346526992928
# Full validation mse:        0.346068824114
# 'Public' validation mse:    0.317470247945
# 'Private' validation mse:   0.359734604285

n_public = 5 # Number of days in public test set
weights=pd.concat([items["perishable"]]) * 0.25 + 1

print("Unweighted validation mse: ", mean_squared_error(
    y_val, np.array(val_pred).transpose()))
print("Full validation mse:       ", mean_squared_error(
    y_val, np.array(val_pred).transpose(), sample_weight=weights))
print("'Public' validation mse:   ", mean_squared_error(
    np.array(y_val)[:,:n_public], np.array(val_pred).transpose()[:,:n_public]))
print("'Private' validation mse:  ", mean_squared_error(
    np.array(y_val)[:,n_public:], np.array(val_pred).transpose()[:,n_public:]))


print("Preparing dataset again for training until 10Aug...")
# add 3 more moving periods
t2017 = date(2017, 5, 17)
X_l, y_l = [], []
for i in range(11):
    print('step:',i+1)
#     if i>4:
#         t2017 = date(2016, 7, 13)
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(
        t2017 + delta
    )
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l

X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

print("Training and predicting models...")
params = {
    'num_leaves': 2**8 - 1,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.025,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'num_threads': 6
}

MAX_ROUNDS = 1000
test_pred_full = []
cate_vars = []
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i+1))
    print("=" * 50)
    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=pd.concat([items["perishable"]] * 11) * 0.25 + 1
    )
    bst = lgb.train(
        params, dtrain, num_boost_round=n_rounds[i] or MAX_ROUNDS,
        verbose_eval=50
    )
    test_pred_full.append(bst.predict(
        X_test, num_iteration=n_rounds[i] or MAX_ROUNDS))


print("Making submission...")
y_test = np.array(test_pred_full).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
# submission.to_csv('lgb_cv_0.3460_full.csv', float_format='%.4f', index=None)
submission.to_csv('lgb_cv_0.3442_full.csv', float_format='%.4f', index=None)
