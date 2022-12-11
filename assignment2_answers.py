# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:09:55 2022

@author: TharindaArachchi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def read_external_files(filename):
    '''
    Read an external file and load into a dataframe, create another dataframe 
    by transposing original one

    Parameters
    ----------
    filename : external file name with extension

    Returns
    -------
    a dataframe and it's transpose

    '''
    #Look for the extension of the file 
    splitFileName = os.path.splitext(filename)
    fileExtension = splitFileName[1]
    
    #World bank data is in csv, xml and excel formats only. Based on those formats reading 
    #the files into dataframe
    if (fileExtension == '.csv'):
        df_climt_chg = pd.read_csv(filename, skiprows=4)
        df_climt_chg_tp = df_climt_chg.set_index('Country Name').transpose()
    ##TO DO :: XML read file
    elif (fileExtension == '.xls'):
        df_climt_chg = pd.read_excel(filename, skiprows=3)
        df_climt_chg_tp = df_climt_chg.set_index('Country Name').transpose()
    else:
        raise Exception("Invalid File Format")
    return df_climt_chg, df_climt_chg_tp
   
#Executing the function to load external file to dataframe     
df_climt_chg, df_climt_chg_tp = read_external_files('API_19_DS2_en_csv_v2_4700503.csv')    

#Select few countries representing all continets from the dataset
countries = ['United Kingdom', 'India', 'Japan', 'China', 'Australia',
             'South Africa', 'United States', 'United Arab Emirates', 'Brazil', 'Italy']
df_countries = df_climt_chg[df_climt_chg['Country Name'].isin(countries)]

#Select records for specific indicators
indecators = ['Urban population', 'Population, total', 'CO2 emissions (kt)']
df_cntry_ind = df_countries[df_countries['Indicator Name'].isin(indecators)]

#Select only columns after year 1990
df_cntry_yrs = df_cntry_ind[['Country Name', 'Indicator Name', '1990', '1991', '1992',	
                             '1993', '1994', '1995', '1996', '1997', '1998', '1999',
                             '2000', '2001', '2002', '2003', '2004', '2005', '2006',
                             '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                             '2014', '2015', '2016', '2017', '2018', '2019', '2020']]
#Replace NaN values with mean
df_cntry_yrs['2020'].fillna(value=df_cntry_yrs['2020'].mean(), inplace=True)

#Plot bar chart against indecator Urban population
df_urbn_pop = df_cntry_yrs[df_cntry_yrs['Indicator Name'] == 'Urban population']
plt.figure()
plt.style.use('ggplot')
# plotting graph
df_urbn_pop.plot(x="Country Name", y=["1990", "1991"], kind='bar')