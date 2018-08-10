#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 10:31:52 2018

@author: InJuly
"""

#forecast
import pandas as pd
import matplotlib.pyplot as plt
import fbprophet

#data 
df = pd.read_csv('1_connection_testdata.csv')


#cleaning, manipulating and setting up data for forecasting
df['Sale Rate (%)'] = round(df['Ad Impressions']/df['Ad Requests']*100,2)
df['Adj Revenue ($)'] =  df['Sale Rate (%)'] * df['CPM ($)'] /10

ts_cpm = pd.DataFrame()
ts_cpm['Time'] = df['Date']
ts_cpm['Ad Requests'] = df['Ad Requests']

#set up time index 

#ts_cpm['Time']=pd.to_datetime(ts_cpm['Time'])
#ts_cpm = ts_cpm.set_index('Time')

#plot cpm
#plt.plot(ts_cpm.index, ts_cpm['CPM'])
#plt.title('CPM')
#plt.ylabel('CPM ($)');
#plt.show()



# Prophet requires columns ds (Date) and y (value)
ts_cpm = ts_cpm.rename(columns={'Time': 'ds', 'Ad Requests': 'y'})

# Make the prophet model and fit on the data
#scale is how sensitive to change
#wweekly_seasonality=True
cpm_prophet = fbprophet.Prophet(changepoint_prior_scale=1)
cpm_prophet.fit(ts_cpm)

# Make a future dataframe for 2 weeks
cpm_forecast = cpm_prophet.make_future_dataframe(periods= 5 * 1, freq='D')
# Make predictions
cpm_forecast = cpm_prophet.predict(cpm_forecast)

#blue dot is actual cpm, blue line is forecasting trend
#shaded area is uncertainty
#(anything in this region would be reasonable)
cpm_prophet.plot(cpm_forecast, xlabel = 'Date', ylabel = 'Ad Requests')
plt.title('CPM forecast');






