import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import sklearn.preprocessing as preprocessing
import pandas as pd
import numpy as np
from GitHub.ADA_Project.LoadingData import data
import seaborn as sns
import matplotlib.pyplot as plt
import csv

# Loading the data class
Data = data()
Data.loadData()

# Computing the first cases
nonZero = []
for country in Data.y.index:
    for i, date in enumerate(Data.y.columns):
        if Data.y.loc[country, date] != 0:
            index = Data.y.columns.get_loc(date)
            nonZero.append(index)
            break


# Final Data Preparation
FeaturesContinent = Data.X['Continent']
Data.X = Data.X.drop(['Continent'], axis=1)
enc = preprocessing.LabelEncoder()
encC = enc.fit(FeaturesContinent)
FeaturesContinent = encC.transform(FeaturesContinent)
ohe = preprocessing.OneHotEncoder()
encodedC = ohe.fit(FeaturesContinent.reshape(-1, 1))
FeaturesContinent = encodedC.transform(FeaturesContinent.reshape(-1, 1)).toarray()
Features = np.concatenate([FeaturesContinent], axis=1)
Features = np.concatenate([Features, np.array(Data.X[['GDP', 'Gini', 'Dem', 'Pop', 'Health', 'Child', 'Dens', 'Trade',
                                                      'Political Stability', 'GOV', 'Distance']])], axis=1)
X = pd.DataFrame(Features, index=Data.y.index, columns=['South America', 'Europe', 'Oceania', 'North America', 'Asia',
                                                        'Africa', 'GDP', 'Gini', 'Dem', 'Pop', 'Health', 'Child',
                                                        'Dens', 'Trade', 'Political Stability', 'GOV', 'Distance'])

nonZero = pd.DataFrame(np.array(nonZero), index=Data.y.index, columns={'First Case'})
X = X.join(nonZero)

del nonZero, i, index, country, date, Features, FeaturesContinent, enc, encC, encodedC, ohe

# Loading optimise parameters

with open('Time.csv') as csv_file:
    reader = csv.reader(csv_file)
    T = dict(reader)

with open('Predictions.csv') as csv_file:
    reader = csv.reader(csv_file)
    Ypred = dict(reader)

with open('True.csv') as csv_file:
    reader = csv.reader(csv_file)
    Ytrue = dict(reader)

with open('W.csv') as csv_file:
    reader = csv.reader(csv_file)
    W = dict(reader)

Lmax = []
Ks = []
X0s = []
for country in Data.y.index:
    p = W["Parameters " + country].replace('[', '').replace(']', '').split()
    p = pd.DataFrame(p).values
    ws = []
    for j in p:
        ws.append(float(j))
    Lmax.append(ws[0])
    Ks.append(ws[1])
    X0s.append(ws[2])

# We format the true value of paremeters in pandas DataFrame
Lmax = pd.DataFrame(Lmax, columns=['L'], index=Data.y.index)
Ks = pd.DataFrame(Ks, columns=['K'], index=Data.y.index)
X0s = pd.DataFrame(X0s, columns=['x0'], index=Data.y.index)
OptiParam = Lmax.join([Ks, X0s])

del Lmax, Ks, X0s, T, Ytrue, Ypred, W, reader, csv_file

# ***********************************************************
# Neural Net Implementation
# **********************************************************

# Splitting the data
Data.split(percentSplit=0.8)

# Exploring the joint distribution
sns.pairplot(X.loc[Data.TrainSplit, ['GDP', 'Gini', 'Pop', 'Health', 'First Case']], diag_kind="kde")
plt.show()

# Normalising the data
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[X.shape[1]]),
        layers.Dense(64, activation='relu'),
        layers.Dense(2)
    ])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])

    return model

NN = build_model()

EPOCHS = 10000
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = NN.fit(
  X.loc[Data.TrainSplit, :].astype(float), OptiParam.loc[Data.TrainSplit, ['K', 'L']].astype(float),
  epochs=EPOCHS, validation_split=0.2, verbose=0,
  callbacks=[early_stop, tfdocs.modeling.EpochDots()])


plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plt.subplot(121)
plotter.plot({'Basic': history}, metric="mae")
plt.ylabel('MAE [MPG]')
plt.subplot(122)
plotter.plot({'Basic': history}, metric="mse")
plt.ylabel('MSE [MPG^2]')
plt.show()

loss, mae, mse = NN.evaluate(X.loc[Data.TestSplit, :].astype(float),
                             OptiParam.loc[Data.TestSplit, ['K', 'L']].astype(float), verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))
