import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import geocoder
import folium
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime
from pandas.tseries.offsets import MonthEnd
import ssl

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv("./zhvi.csv")
df = df.dropna()

g = geocoder.arcgis('78258')

def givemap(city, month, year):
    pd.set_option('mode.chained_assignment', None)

    df = pd.read_csv("./zhvi.csv")
    df = df.dropna()
    
    point = df.loc[df['City'].str.contains(str(city))]

    lats = []
    longs = []
    for i in point['RegionName'].to_list():
        if len(str(i)) < 5:
            j = '0'+str(i)
        else:
            j = str(i)
        g = geocoder.arcgis(j)
        lats.append(g.lat)
        longs.append(g.lng)
    point['Latitude'] = lats
    point['Longitude'] = longs
    
    point = point.drop(columns=['RegionID', 'SizeRank', 'RegionType', 'StateName', 'State', 'City', 'CountyName'])
    temp_cols=point.columns.tolist()
    new_cols=temp_cols[-2:] + temp_cols[:-2]
    a=point[new_cols]
    b = a.melt(id_vars=["RegionName", "Metro", "Latitude", "Longitude"], 
            var_name="Date", 
            value_name="Value")
    b['Date'] = pd.to_datetime(b['Date'])
    zoom_index = [b['Latitude'].iloc[0], b['Longitude'].iloc[0]]
    
    time = pd.to_datetime((str(month)+str(year))) + MonthEnd(1)
    filtered = b.loc[b['Date']==time]
    f = filtered.set_index('Date')
    
    m = folium.Map(location=zoom_index, zoom_start=11)
    y = filtered[['Latitude', 'Longitude', 'Value']]
    hm = HeatMap(y,radius=25,min_opacity=.7)
    hm.add_to(m)
    return m

givemap("San Jose", "June", 2017)
