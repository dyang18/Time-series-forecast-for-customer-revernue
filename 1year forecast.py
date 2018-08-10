#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 13:56:09 2018

@author: InJuly
"""
#1 year data of forecast 
#forecast
import pandas as pd
import matplotlib.pyplot as plt
import fbprophet

#data 
year_df = pd.read_csv('~/Desktop/libring/libring_edge_ad_network_and_mediation_dashboard_1531237798.csv')


#cleaning, manipulating and setting up data for forecasting
year_df = year_df[['Date','Ad Format','Ad Impressions','CPM ($)','Ad Revenue ($)']]
year_df = year_df.groupby(['Date'],as_index=False)['Ad Impressions','Ad Revenue ($)'].sum()
year_df['CPM'] = year_df['Ad Revenue ($)'] / year_df['Ad Impressions'] *1000

year_df = year_df[['Date','CPM']]

# Prophet requires columns ds (Date) and y (value)
year_df = year_df.rename(columns={'Date': 'ds', 'CPM': 'y'})

# Make the prophet model and fit on the data
#scale is how sensitive to change
year_prophet = fbprophet.Prophet(changepoint_prior_scale=1000)
year_prophet.fit(year_df)

# Make a future dataframe for 2 weeks
year_forecast = year_prophet.make_future_dataframe(periods= 7 * 4, freq='D')
# Make predictions
year_forecast = year_prophet.predict(year_forecast)

#blue dot is actual cpm, blue line is forecasting trend
#shaded area is uncertainty
#(anything in this region would be reasonable)
year_prophet.plot(year_forecast, xlabel = 'Date', ylabel = 'CPM ($)')
plt.title('CPM forecast');








