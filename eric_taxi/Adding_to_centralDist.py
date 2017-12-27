#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:40:27 2017

@author: eric
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline  








df=pd.read_csv('last_FiveMonth.csv')




def get_dist_StartToCentral(df):
       data=df.loc[df['start_longitude']!='no information']
       data=data.loc[data['start_latitude']!='no information']
       lon_diff = np.abs(data.start_longitude.astype(float)-(-8.62195))*np.pi/360.0
       lat_diff = np.abs(data.start_latitude.astype(float)-41.162142)*np.pi/360.0
       a = np.sin(lat_diff)**2 + np.cos((-8.62195)*np.pi/180.0) * np.cos(data.start_longitude.astype(float)*np.pi/180.0) * np.sin(lon_diff)**2  
       d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))   # in km
       return(d)



df_start_to_central=get_dist_StartToCentral(df)

df_start_to_central = pd.DataFrame({ 'StartDistToCentral': df_start_to_central})

df=pd.concat([df_start_to_central,df],axis=1)

df['StartDistToCentral']=df['StartDistToCentral'].fillna('no information')



def get_dist_EndToCentral(df):
       data=df.loc[df['end_longitude']!='no information']
       data=data.loc[data['end_latitude']!='no information']
       lon_diff = np.abs(data.end_longitude.astype(float)-(-8.62195))*np.pi/360.0
       lat_diff = np.abs(data.end_latitude.astype(float)-41.162142)*np.pi/360.0
       a = np.sin(lat_diff)**2 + np.cos((-8.62195)*np.pi/180.0) * np.cos(data.end_longitude.astype(float)*np.pi/180.0) * np.sin(lon_diff)**2  
       d = 2*6371*np.arctan2(np.sqrt(a), np.sqrt(1-a))   # in km
       return(d)





df_end_to_central=get_dist_EndToCentral(df)

df_end_to_central = pd.DataFrame({ 'EndDistToCentral': df_end_to_central})

df=pd.concat([df_end_to_central,df],axis=1)

df['EndDistToCentral']=df['EndDistToCentral'].fillna('no information')

last_FiveMonth=df 
last_FiveMonth.to_csv('last_FiveMonth.csv', index=False)



