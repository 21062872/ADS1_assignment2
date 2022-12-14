# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 12:09:55 2022

@author: TharindaArachchi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sb
import scipy.stats as stats


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
    
    #World bank data is in csv, excel formats. Based on those formats reading the files into dataframe
    if (fileExtension == '.csv'):
        df_climt_chg = pd.read_csv(filename, skiprows=4)
        df_climt_chg_tp = df_climt_chg.set_index('Country Name').transpose()
    elif (fileExtension == '.xls'):
        df_climt_chg = pd.read_excel(filename, skiprows=3)
        df_climt_chg_tp = df_climt_chg.set_index('Country Name').transpose()
    else:
        raise Exception("Invalid File Format")
    return df_climt_chg, df_climt_chg_tp

def calculateAutoPct(values):
    '''Calculate percentage of portion
        Used for labeling the pie chart'''
    def autopct(pct):
        return '{p:.2f}%'.format(p=pct)
    return autopct

def calculateStats(dataset):
    '''
    Read dataframe and calculate statistics

    Parameters
    ----------
    dataframe

    Returns
    -------
    calculated statistics

    '''    
    #Get numeric columns only
    df_numeric = dataset.select_dtypes(include=np.number)
    vmax = np.max(df_numeric, axis=1)
    vmin = np.min(df_numeric, axis=1)
    vmean = np.mean(df_numeric, axis=1)
    return vmax, vmin, vmean

#Executing the function to load external file to dataframe     
df_climt_chg, df_climt_chg_tp = read_external_files('API_19_DS2_en_csv_v2_4700503.csv')    

#Select few countries representing all continets from the dataset
countries = ['United Kingdom', 'India', 'Japan', 'China', 'Korea, Rep.',
             'South Africa', 'United States', 'Korea, Rep.', 'Germany']
df_countries = df_climt_chg[df_climt_chg['Country Name'].isin(countries)]

#Select records for specific indicators
indecators = ['CO2 emissions (kt)', 'Urban population', 'Population, total', 'Forest area (% of land area)',
              'Arable land (% of land area)', 'Total greenhouse gas emissions (kt of CO2 equivalent)', 
              'Methane emissions (kt of CO2 equivalent)', 'Mortality rate, under-5 (per 1,000 live births)',
              'Agricultural land (sq. km)']
df_cntry_ind = df_countries[df_countries['Indicator Name'].isin(indecators)]
#Select only columns after year 1990
df_cntry_yrs = df_cntry_ind.loc[:,['Country Name', 'Indicator Name', '1990', '1991', '1992',	
                                    '1993', '1994', '1995', '1996', '1997', '1998', '1999',
                                    '2000', '2001', '2002', '2003', '2004', '2005', '2006',
                                    '2007', '2008', '2009', '2010', '2011', '2012', '2013',
                                    '2014', '2015', '2016', '2017', '2018', '2019', '2020']]

#Replace NaN values with preceding row value
df_cntry_yrs = df_cntry_yrs.fillna(method='ffill', axis=1)

#Select data in 5 year intervals 
df_cntry_ind_hdr = df_cntry_yrs.iloc[:,[0,1]]
df_yrs_five_intr = df_cntry_yrs.iloc[:,2::5]
df_fnl = pd.concat([df_cntry_ind_hdr,df_yrs_five_intr], axis=1)

#Rename countries with shortnames for better clarity of labels
df_fnl.replace("United Kingdom", "UK", inplace=True)
df_fnl.replace("United States", "USA", inplace=True)
df_fnl.replace("South Africa", "SA", inplace=True)
df_fnl.replace("Korea, Rep.", "Korea", inplace=True)

''' Plot: 1
    Plot type: Bar chart
    Plot name: Urban Population Over Time'''
#Plot bar chart against indicator Urban population
df_urbn_pop = df_fnl[df_fnl['Indicator Name'] == 'Urban population']

# plotting graph
plt.figure(figsize=(8, 6), dpi=80)
plt.style.use('ggplot')
# plot grouped bar chart
df_urbn_pop.plot(x='Country Name',
                kind='bar',
                stacked=False,
                title='Urban Population Over Time')
# labeling the graph
plt.xlabel('Country')
plt.ylabel('Number of population')
plt.legend(title ="Years", bbox_to_anchor =(0.8, 1, 0.3, 0))
plt.show()

''' Plot: 2
    Plot type: Bar chart
    Plot name: CO2 emission Over Time'''
#Plot bar chart against indicator CO2 emmision
df_co2_emsn = df_fnl[df_fnl['Indicator Name'] == 'CO2 emissions (kt)']

# plotting graph
plt.figure(figsize=(8, 6), dpi=80)
plt.style.use('ggplot')
# plot grouped bar chart
df_co2_emsn.plot(x='Country Name',
                kind='bar',
                stacked=False,
                title='CO2 Emission Over Time')
# labeling the graph
plt.xlabel('Country')
plt.ylabel('CO2 emission')
plt.legend(title ="Years", loc='upper right', bbox_to_anchor =(0.8, 1, 0.3, 0))
plt.show()

''' Plot: 3
    Plot type: Line chart
    Plot name: Forest Area'''
# create a new figure
plt.figure(figsize=(8,5))
# add a stylesheet
plt.style.use('ggplot')
# adding a title to the plot
plt.title('Forest Area')
# plot the grapgh
df_ln_plot = df_fnl[df_fnl['Indicator Name'] == 'Forest area (% of land area)']
df_ln_plot = df_ln_plot.loc[:,['Country Name', '1990', '1995', '2000', '2005', '2010', '2015', '2020']]
df_ln_plot_tp = df_ln_plot.set_index('Country Name').transpose()
df_ln_plot_tp['Year'] = df_ln_plot_tp.index
# plot the grapgh
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.China,  label = 'China')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.India,  label = 'India')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.USA,  label = 'USA')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.UK,  label = 'UK')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.Germany,  label = 'Germany')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.SA,  label = 'SA')
plt.xticks(df_ln_plot_tp.Year)
# labeling the graph
plt.xlabel('Year')
plt.ylabel('SQ(KM)')
plt.legend(loc = 'lower right', title ="Countries", bbox_to_anchor =(0.8, 0, 0.3, 0))
plt.show()

''' Plot: 4
    Plot type: Line chart
    Plot name: Agricultural land (sq. km)'''
# create a new figure
plt.figure(figsize=(8,5))
# add a stylesheet
plt.style.use('ggplot')
# adding a title to the plot
plt.title('Agricultaral Area')
# plot the grapgh
df_ln_plot = df_fnl[df_fnl['Indicator Name'] == 'Agricultural land (sq. km)']
df_ln_plot = df_ln_plot.loc[:,['Country Name', '1990', '1995', '2000', '2005', '2010', '2015', '2020']]
df_ln_plot_tp = df_ln_plot.set_index('Country Name').transpose()
df_ln_plot_tp['Year'] = df_ln_plot_tp.index
# plot the grapgh
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.China,  label = 'China')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.India,  label = 'India')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.USA,  label = 'USA')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.UK,  label = 'UK')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.Germany,  label = 'Germany')
plt.plot(df_ln_plot_tp.Year, df_ln_plot_tp.SA,  label = 'SA')
plt.xticks(df_ln_plot_tp.Year)
# labeling the graph
plt.xlabel('Year')
plt.ylabel('SQ(KM)')
plt.legend(loc = 'lower right', title ="Countries", bbox_to_anchor =(0.8, 0, 0.3, 0))
plt.show()

''' Plot: 4
    Plot type: Pie chart
    Plot name: CO2 emission in 2020'''
df_co2_emsn_pc = df_fnl[df_fnl['Indicator Name'] == 'CO2 emissions (kt)']
df_co2_emsn_pc = df_co2_emsn_pc.loc[:, ['Country Name', '2020']]

# defining color list for pie chart
colors = ("red", "cyan","Yellow", 'green', '#a87d32', '#57094a', '#808dd1', '#a8c236')

# wedge properties
wp = { 'linewidth' : 1, 'edgecolor' : "green" }

# creating plot
plt.figure(figsize=(8,5))
plt.pie(df_co2_emsn_pc['2020'], labels=df_co2_emsn_pc['Country Name'], shadow = True, colors=colors,  \
        startangle = 90, wedgeprops = wp, autopct=calculateAutoPct(df_co2_emsn_pc['2020']))
# adding legend
plt.legend(
              title ="CO2 emission",
              loc ="lower left",
              bbox_to_anchor =(1.1, 0, 0.5, 1)
          )   
plt.tight_layout()
plt.title("CO2 Emission in 2020")
plt.show()

''' Plot: 4
    Plot type: horizontal bar chart
    Plot name: Average Co2 emission per year'''
df_co2_emsn_tot = df_fnl[df_fnl['Indicator Name'] == 'CO2 emissions (kt)']
df_co2_emsn_tot = df_co2_emsn_tot.loc[:,['Country Name', '1990', '1995', '2000', '2005', '2010', '2015', '2020']]
#Sum up figures in each row
df_co2_emsn_tot['Total'] = df_co2_emsn_tot.sum(axis=1, numeric_only=True)
#Find average figures in each row
df_co2_emsn_tot['Avg'] = df_co2_emsn_tot.mean(axis=1, numeric_only=True)
#Sorting dataframe by average column
df_co2_emsn_tot = df_co2_emsn_tot.sort_values(by='Avg')

# create a new figure
plt.figure(figsize=(8,5))
# add a stylesheet
plt.style.use('ggplot')
# plot the grapgh
df_co2_emsn_tot.plot.barh('Country Name', 'Avg', label='Avg CO2 emission',  color=(0.2, 0.4, 0.6, 0.6))
# labeling the graph
plt.ylabel('Country')
plt.xlabel('Co2 emission(kt)')
plt.title('Average Co2 emission(kt) per year')
plt.legend(loc = 'best')
plt.show()

#Derive mathematical statistics
df_stats = df_fnl[df_fnl['Indicator Name'] == 'CO2 emissions (kt)']
chn_mean = df_stats[df_stats['Country Name'] == 'China']

otp = calculateStats(chn_mean)
    
print('\nAverage Co2 emission of China per year is ', otp[2])
print('\nMaximum Co2 emission recorded in China is ', otp[0])
print('\nMinimum Co2 emission recorded in China is ', otp[1])

''' Plot: 5
    Plot type: correlation heatmap
    Plot name: correlation heatmap of USA'''
df_cor = df_fnl[df_fnl['Country Name'] == 'USA']
df_cor = df_cor.loc[:,['Indicator Name', '1990', '1995', '2000', '2005', '2010', '2015', '2020']]
df_cor = df_cor.set_index('Indicator Name').transpose()
df_cor.rename(columns = {'CO2 emissions (kt)':'CO2 emission', 
                        'Urban population':'Urban population', 
                        'Population, total':'Total population', 
                        'Forest area (% of land area)':'Forest area',
                        'Arable land (% of land area)':'Arable area', 
                        'Total greenhouse gas emissions (kt of CO2 equivalent)':'Greenhouse gas emission', 
                        'Methane emissions (kt of CO2 equivalent)':'Methane emission', 
                        'Mortality rate, under-5 (per 1,000 live births)':'Mortality rate (Under 5)'
                        }, inplace = True)
print(df_cor.corr())
# plotting correlation heatmap
dataplot = sb.heatmap(df_cor.corr(), cmap="YlGnBu", annot=True)
# displaying heatmap
plt.show()

#Identify skewness of dataset for USA  
plt.figure()
plt.hist(df_cor['Methane emission'], alpha=0.7, color='#ab34eb', edgecolor='black')
plt.title('Methane emission')
plt.show()

plt.figure()
plt.hist(df_cor['Forest area'], alpha=0.7, color='#ab34eb', edgecolor='black')
plt.title('Forest area')
plt.show()


print('\nSkewness of Methane emission is ', stats.skew(df_cor['Methane emission']))
print('\nSkewness of Forest area is ', stats.skew(df_cor['Forest area']))