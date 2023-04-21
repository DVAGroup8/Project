import pandas as pd
data = pd.read_csv("./Zillow_Airbnb.csv")

grouped = data.groupby(data['City'])

groupdict = {}
for val in set(data['City'].values):
    groupdict[val] = grouped.get_group(val)

listings = []
for key in set(groupdict.keys()):
    listings.append(groupdict[key].iloc[groupdict[key]['Listings'].to_numpy().nonzero()[0]].iloc[0]['Date'])

for key in set(groupdict.keys()):
    groupdict[key]['Date'] = pd.to_datetime(groupdict[key]['Date'], format='%Y-%m-%d')

for key in set(groupdict.keys()):
    groupdict[key] = groupdict[key][groupdict[key]['Date'] >= '2011-12-31']

groupdict['New York, NY']

for key in set(groupdict.keys()):
    groupdict[key] = groupdict[key].drop(columns = ['Unnamed: 0', 'RegionID', 'SizeRank', 'City', 'RegionType', 'StateName'])

rate = pd.read_csv("FEDFUNDS.csv")
cpi = pd.read_csv("CPIAUCSL.csv")
unrate = pd.read_csv("UNRATE.csv")

rate['DATE'] = pd.to_datetime(rate['DATE'], format='%m/%d/%y')
cpi['DATE'] = pd.to_datetime(cpi['DATE'], format='%m/%d/%y')
unrate['DATE'] = pd.to_datetime(unrate['DATE'], format='%m/%d/%y')

for key in set(groupdict.keys()):
    groupdict[key] = groupdict[key].merge(rate, left_on = 'Date', right_on = 'DATE')
    groupdict[key] = groupdict[key].drop(columns = ['DATE', 'DATE'])
    groupdict[key] = groupdict[key].merge(cpi, left_on = 'Date', right_on = 'DATE')
    groupdict[key] = groupdict[key].drop(columns = ['DATE'])
    groupdict[key] = groupdict[key].merge(unrate, left_on = 'Date', right_on = 'DATE')
    groupdict[key] = groupdict[key].drop(columns = ['DATE'])

diffdict = {}
for key in set(groupdict.keys()):
    diffdict[key] = groupdict[key].copy()

diffdict['Dallas, TX']

for key in set(diffdict.keys()):
    diffdict[key]['ZHVI'] = diffdict[key]['ZHVI'].diff()
    diffdict[key]['Listings'] = diffdict[key]['Listings'].diff()
    diffdict[key]['FEDFUNDS'] = diffdict[key]['FEDFUNDS'].diff()
    diffdict[key]['CPIAUCSL'] = diffdict[key]['CPIAUCSL'].diff()
    diffdict[key]['UNRATE'] = diffdict[key]['UNRATE'].diff()
    diffdict[key] = diffdict[key][diffdict[key]['Date'] > '2011-12-31']

reg_xs = {}
reg_ys = {}
for key in set(groupdict.keys()):
    reg_xs[key] = groupdict[key].drop(columns = ['Date', 'ZHVI'])
    reg_ys[key] = groupdict[key]['ZHVI']

diff_xs = {}
diff_ys = {}
for key in set(groupdict.keys()):
    diff_xs[key] = diffdict[key].drop(columns = ['Date', 'ZHVI'])
    diff_ys[key] = diffdict[key]['ZHVI']

import statsmodels.api as sm
reg_models = {}
p_values_reg = {}
R_sq_reg = {}
summaries_reg = {}
for key in set(groupdict.keys()):
    reg_models[key] = sm.OLS(reg_ys[key],reg_xs[key])
    fii = reg_models[key].fit()
    p_values_reg[key] = fii.summary2().tables[1]['P>|t|']
    R_sq_reg[key] = fii.summary2().tables[0][1][6]
    summaries_reg[key] = fii.summary()

import statsmodels.api as sm
diff_models = {}
p_values_diff = {}
R_sq_diff = {}
summaries_diff = {}
for key in set(groupdict.keys()):
    diff_models[key] = sm.OLS(diff_ys[key],diff_xs[key])
    fii = diff_models[key].fit()
    p_values_diff[key] = fii.summary2().tables[1]['P>|t|']
    R_sq_diff[key] = fii.summary2().tables[0][1][6]
    summaries_diff[key] = fii.summary()

summaries_reg

scaled_reg_xs = {}
scaled_reg_ys = {}
scaled_diff_xs = {}
scaled_diff_ys = {}
for key in set(groupdict.keys()):
    scaled_reg_xs[key] = (reg_xs[key]-reg_xs[key].mean())/reg_xs[key].std()
    scaled_reg_ys[key] = (reg_ys[key]-reg_ys[key].mean())/reg_ys[key].std()
    scaled_diff_xs[key] = (diff_xs[key]-diff_xs[key].mean())/diff_xs[key].std()
    scaled_diff_ys[key] = (diff_ys[key]-diff_ys[key].mean())/diff_ys[key].std()

reg_models_scaled = {}
p_values_reg_scaled = {}
R_sq_reg_scaled = {}
summaries_reg_scaled = {}
for key in set(groupdict.keys()):
    reg_models_scaled[key] = sm.OLS(scaled_reg_ys[key],scaled_reg_xs[key])
    fii = reg_models_scaled[key].fit()
    p_values_reg_scaled[key] = fii.summary2().tables[1]['P>|t|']
    R_sq_reg_scaled[key] = fii.summary2().tables[0][1][6]
    summaries_reg_scaled[key] = fii.summary()

diff_models_scaled = {}
p_values_diff_scaled = {}
R_sq_diff_scaled = {}
summaries_diff_scaled = {}
for key in set(groupdict.keys()):
    diff_models_scaled[key] = sm.OLS(scaled_diff_ys[key],scaled_diff_xs[key])
    fii = diff_models_scaled[key].fit()
    p_values_diff_scaled[key] = fii.summary2().tables[1]['P>|t|']
    R_sq_diff_scaled[key] = fii.summary2().tables[0][1][6]
    summaries_diff_scaled[key] = fii.summary()

summaries_diff_scaled

scaled_reg_xs['Dallas, TX']

## Time Series

groupdict['Dallas, TX']['Date'].min(), groupdict['Dallas, TX']['Date'].max()

groupdict['Dallas, TX']

tsdict = {}
for key in set(groupdict.keys()):
    tsdict[key] = groupdict[key].set_index('Date')
tsdict['Dallas, TX']

y = tsdict['Dallas, TX']['ZHVI']
y

import matplotlib.pyplot as plt
for key in set(groupdict.keys()):
    y = tsdict[key]['ZHVI']
    y.plot(figsize=(15,9),label = key)
plt.legend(loc="upper left")
plt.title("Time Series Plot of Average Zillow Home Value Index (ZHVI) in Each Key City")
plt.show()

timeseries = tsdict['Chicago, IL']['ZHVI']
timeseries.rolling(12).mean().plot(label='12 Month Rolling Mean')
timeseries.rolling(12).std().plot(label='12 Month Rolling Std')
timeseries.plot()
plt.legend() 

from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
y = tsdict['Chicago, IL']['ZHVI']
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#positive trend, seasonality, noisy at end of sample (for multiple cities)

from statsmodels.tsa.stattools import adfuller
for key in set(tsdict.keys()):
    test_result = adfuller(tsdict[key]['ZHVI'])
    print ('ADF Test:')
    labels = ['ADF Statistic','p-value','No. of Lags Used','Number of Observations Used']

    for value,label in zip(test_result,labels):
        print (label+': '+str(value))
    if test_result[1] <= 0.05:
        print ("Reject null hypothesis and data is stationary")
    else:
        print ("Fail to reject H0 thereby data is non-stationary ")
#all are nonstationary so we can take the difference to make the data stationary

tsdiffdict = {}
for key in set(tsdict.keys()):
    tsdiffdict[key] = tsdict[key]['ZHVI'] - tsdict[key]['ZHVI'].shift(1)
    test_result = adfuller(tsdiffdict[key].dropna())
    print ('ADF Test:')
    labels = ['ADF Statistic','p-value','No. of Lags Used','Number of Observations Used']

    for value,label in zip(test_result,labels):
        print (label+': '+str(value))
    if test_result[1] <= 0.05:
        print ("Reject null hypothesis and data is stationary")
    else:
        print ("Fail to reject H0 thereby data is non-stationary ")

import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

setup = []
y = tsdict["Chicago, IL"]['ZHVI']
min_aic = 100000
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

            results = mod.fit()
            
            if results.aic < min_aic:
                min_aic = results.aic
                setup = [param, param_seasonal, results.aic]
        except:
            continue

setup

from statsmodels.tools.sm_exceptions import ValueWarning
import warnings
warnings.simplefilter('ignore', ValueWarning)
setup = {}
for key in set(tsdict.keys()):
    y = tsdict[key]['ZHVI']
    min_aic = 100000
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)

                results = mod.fit()

                if results.aic < min_aic:
                    min_aic = results.aic
                    setup[key] = param, param_seasonal, min_aic
            except:
                continue


setup

ts_models = {}
p_values_ts = {}
R_sq_ts = {}
summaries_ts = {}
fii = {}
y = {}
for key in set(setup.keys()):
    y[key] = tsdict[key]['ZHVI']
    ts_models[key] = sm.tsa.statespace.SARIMAX(y[key], order=setup[key][0], seasonal_order=setup[key][1], enforce_stationarity=False, enforce_invertibility=False)
    fii[key] = ts_models[key].fit()
    summaries_ts[key] = fii[key].summary()

summaries_ts

fii['Chicago, IL'].plot_diagnostics(figsize=(16, 8))
plt.show()
#checking if there is unusual behavior.
#data appears to have constant variance, relatively normal distribution and no trend over time (all key for a good TS model)

import numpy as np
#only one city's prediction
pred = fii['Chicago, IL'].get_prediction(start=pd.to_datetime('2022-01-31'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['Chicago, IL']['2019-01-31':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('ZHVI')
plt.legend()
plt.show()

import numpy as np
for key in set(fii.keys()):
    pred = fii[key].get_prediction(start=pd.to_datetime('2022-01-31'), dynamic=False).predicted_mean
    y_truth = y[key]['2022-01-31':]
    mse = ((pred - y_truth) ** 2).mean()

    #print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
    print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

import numpy as np
#all cities predictions
pred = {}
y_forecasted = {}
mse = {}
zhvi_prediction = pd.DataFrame()
mse = pd.DataFrame()
for key in set(fii.keys()):
    pred = fii[key].get_prediction(start=pd.to_datetime('2022-01-31'), dynamic=False).predicted_mean
    zhvi_prediction[key] = pred
    
    y_truth = y[key]['2022-01-31':]
    mse[key] = ((pred - y_truth) ** 2).mean()

pred = fii['Dallas, TX'].get_prediction(start=pd.to_datetime('2022-01-31'), dynamic=False).predicted_mean
pred

zhvi_prediction

zhvi_prediction.to_csv("zhvi_predictions.csv")

