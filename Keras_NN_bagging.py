
# importing all libraries
from datetime import date, timedelta
import pandas as pd
import numpy as np
import random as rn
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, PReLU
from keras.layers.normalization import BatchNormalization
from keras.layers import LSTM
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers


# reading data files

df_train = pd.read_csv(
    'train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
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

df_2017 = df_train.loc[df_train.date>=pd.datetime(2016,8,1)]

# transforming data into wide format for deriving features based on past sales
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

# helper functions for preparing data for modelling
def days_since_payment(d):
    payday = 1 if d.day < 15 else 15
    return (d - date(d.year, d.month, payday)).days

def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def get_timespan_dow(df, dt, months=1, days=1, freq='7D'):
    d1 = pd.date_range(end=dt-relativedelta(days=1), start=dt - relativedelta(days=days, months=months), freq=freq)
    return df.iloc[:, df.columns.isin(d1)]

def prepare_dataset(t2017, is_train=True):
    X = pd.DataFrame({
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
        "promo_1_2017": get_timespan(promo_2017, t2017, 1, 1).sum(axis=1).values,
        "promo_3_2017": get_timespan(promo_2017, t2017, 3, 3).sum(axis=1).values,
        "promo_7_2017": get_timespan(promo_2017, t2017, 7, 7).sum(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values
    })
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_7_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 49-i, 7, freq='7D').mean(axis=1).values
        X['mean_10_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').mean(axis=1).values
        X['mean_15_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 105-i, 15, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values

    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X

# preparing dataset based on moving windows
print("Preparing dataset...")
t2017 = date(2017, 5, 31)
X_l, y_l = [], []
for i in range(6):
    print("step",i+1)
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
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)


val_pred = []
test_pred = []
sample_weights=np.array( pd.concat([items["perishable"]] * 6) * 0.25 + 1 )
n_bags = 10
for i in range(16):
    N_EPOCHS = 30
    bagged_val = np.zeros(y_val.shape[0])
    bagged_test = np.zeros(X_test.shape[0])
    y = y_train[:, i]
    xv = np.array(X_val)
    yv = y_val[:, i]
    for j in range(n_bags):
        print("=" * 50)
        print("Step %d" % (i+1), "Bag %d" % (j+1))
        print("=" * 50)

        model = Sequential()
        model.add(Dense(64, input_shape=(X_train.shape[1],),
                        kernel_regularizer=regularizers.l2(0.0001)))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(loss = 'mse', optimizer='adam', metrics=['mse'])
        callbacks = [EarlyStopping(patience=5, verbose=1, min_delta=1e-4)]

        model.fit(np.array(X_train), y, batch_size = 128, epochs = N_EPOCHS, verbose=1, shuffle= False,
                       sample_weight=sample_weights, validation_data=(xv,yv), callbacks=callbacks)

        val_temp = model.predict(np.array(X_val))[:,0]
        bagged_val += val_temp
        test_temp = model.predict(np.array(X_test))[:,0]
        bagged_test += test_temp

    bagged_val /= n_bags
    bagged_test /= n_bags
    val_pred.append(bagged_val)
    test_pred.append(bagged_test)

n_public = 5 # Number of days in public test set
weights=pd.concat([items["perishable"]]) * 0.25 + 1
print("Unweighted validation mse: ", mean_squared_error(
    y_val, np.array(val_pred).transpose()))
print("Full validation mse:       ", mean_squared_error(
    y_val, np.array(val_pred).transpose(), sample_weight=weights))
print("'Public' validation mse:   ", mean_squared_error(
    y_val[:,:n_public], np.array(val_pred).transpose()[:,:n_public],
    sample_weight=weights))
print("'Private' validation mse:  ", mean_squared_error(
    y_val[:,n_public:], np.array(val_pred).transpose()[:,n_public:],
    sample_weight=weights))

# writing predicted test file for submission
print("Making submission...")
df_test = pd.read_csv('test.csv',
                      usecols=[0, 1, 2, 3, 4],
                      dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(
    ['store_nbr', 'item_nbr', 'date'])

y_test = np.array(test_pred).transpose()

df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0).reset_index()
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.loc[~submission.item_nbr.isin(item_nbr_u),'unit_sales']=0
submission = submission[["id", "unit_sales"]]
