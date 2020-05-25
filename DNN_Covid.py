from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
from sklearn import preprocessing
import pandas as pd
import numpy as np
from GitHub.ADA_Project.LoadingData import data
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

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
                                                      'Distance', 'Political Stability', 'GOV']])], axis=1)
X = pd.DataFrame(Features, index=Data.y.index, columns=['South America', 'Europe', 'Oceania', 'North America', 'Asia',
                                                        'Africa', 'GDP', 'Gini', 'Dem', 'Pop', 'Health', 'Child',
                                                        'Dens', 'Trade', 'Distance', 'Political Stability', 'GOV'])

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

del Lmax, Ks, X0s, T, Ytrue, Ypred, W, reader, csv_file, country, j, p, ws

# ***********************************************************
# Neural Net Implementation
# **********************************************************

# Splitting the data
Data.split(percentSplit=0.8)

# Exploring the joint distribution
sns.pairplot(X.loc[Data.TrainSplit, ['GDP', 'Gini', 'Pop', 'Health', 'First Case']], diag_kind="kde")
plt.show()

for indicators in ['GDP', 'Gini', 'Dem', 'Pop', 'Health', 'Child', 'Dens', 'Trade', 'Distance', 'Political Stability', 'GOV']:
    X.loc[:, indicators] = (X.loc[:, indicators].astype(float) - np.mean(X.loc[:, indicators].astype(float)))/np.std(X.loc[:, indicators].astype(float))

# Normalising the data
def build_model():
    model = keras.Sequential([
        layers.Dense(18, activation='relu', input_shape=[X.shape[1]]),
        layers.Dense(6, activation='softmax'),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])

    return model

def fit_model(Output):

    NN = build_model()
    NN_Predict = np.zeros([12, 3])
    count = 0
    EPOCHS = 100000
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    for output in Output:

        history = NN.fit(
          X.loc[Data.TrainSplit, :].astype(float), OptiParam.loc[Data.TrainSplit, [output]].astype(float),
          epochs=EPOCHS, validation_split=0.3, verbose=0,
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
                                     OptiParam.loc[Data.TestSplit, [output]].astype(float), verbose=2)
        print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

        NN_Predict[:, count] = NN.predict(X.loc[Data.TestSplit, :].astype(float)).reshape(12,)

        count += 1

    return NN_Predict

NN_prediction = fit_model(['K', 'L', 'x0'])

def Rsquared(true, false):
    R2 = 0
    for coefficient in range(true.shape[1]):
        print(coefficient)
        a1 = sum((true.iloc[:, coefficient] - false[:, coefficient])**2)
        print(a1)
        a2 = sum((true.iloc[:, coefficient] - np.mean(true.iloc[:, coefficient]))**2)
        print(a2)
        r = 1 - a1/a2
        print(r)
        R2 = R2 + r
    return R2/3

Rsquared_model = Rsquared(OptiParam.loc[Data.TestSplit, :], NN_prediction)



def curve_pred(Parameter):
    x = OptiParam.loc[Data.TestSplit, 'x0']
    ypred = np.zeros((len(Parameter[0]), 151))
    for i in range(len(Parameter[0])):
        for j in range(151):
            ypred[i, j] = Parameter[0][i]/(1+np.exp(Parameter[1][i]*(x[i]-j)))
    return ypred

NN_Pred = curve_pred(NN_prediction)

# Rescaling the true cases
for country in Data.y.index:
    Data.y.loc[country, :] = Data.y.loc[country, :]/Data.X.loc[country, 'Pop']

def R2(ytrue, ypred):

    r = 0

    for count, country in enumerate(Data.TestSplit):
        print(country)
        y_true = ytrue.loc[country]
        y_true = y_true.iloc[X.loc[country, 'First Case']:len(y_true)]

        y_pred = ypred[count, 0:len(y_true)]
        print(len(y_pred))
        print(len(y_true))
        r = r + 1 - sum((y_pred - y_true)**2)/sum((y_true - np.mean(y_true))**2)
        print(r)
    return r/len(Data.TestSplit)

rsquaredNN = R2(Data.y, NN_Pred)



# ***********************************************************
# PCA Implementation
# **********************************************************

IndexTrainNum = []
for count, country in enumerate(Data.TrainSplit):
    IndexTrainNum.append(Data.y.index.get_loc(country))

IndexTestNum = []
for count, country in enumerate(Data.TestSplit):
    IndexTestNum.append(Data.y.index.get_loc(country))

pca = PCA(n_components=2)
pca.fit(np.transpose(X))
LR = LinearRegression()
LR.fit(np.transpose(pca.components_[:, IndexTrainNum]), OptiParam.iloc[IndexTrainNum, :])

