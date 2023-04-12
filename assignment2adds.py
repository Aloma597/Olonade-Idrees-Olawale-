# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 00:40:30 2023

@author: Olonade
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

# tokenizing error was solved by sep =';'
df = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5069170.csv', skiprows = (4))
#def warming(df):
    
    #return df
#warming(df)
#print(df)

#print(df.head(5))
    
#df = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5069170.csv', skiprows = (4))
#def warming_practice(df):
    


def warming(data):
    missing_value= ['Na','NaN','na','np.nan']
    df = pd.read_csv('API_EN.ATM.CO2E.KT_DS2_en_csv_v2_5069170.csv', skiprows = (3), na_values = missing_value)
    df_transpose = df.transpose()
    
    #Handling missing problem
    df_missig_value = df.isnull().any()
    print('df missing value', df_missig_value)
    #visual_df = sns.heatmap(df.isnull(), yticklabels = False, annot = True)
    #print(visual_df)
    df_error = df.isnull().sum()
    print(df_error)
    
    #Handling missing value for df_transpose problem
    df_transpose_missig_value = df_transpose.isnull().any()
    print('transpose missing value', df_transpose_missig_value)
   
    #visual_trans = sns.heatmap(df.isnull(), yticklabels = False, annot = True)
    #print(visual_trans)
    df_trans_error = df_transpose.isnull().sum()
    print(df_trans_error)
    
    # removing the missing values through interpolation
    df.fillna(0)
    aloma = ['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969',
         '1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','2020','2021']
    for i in aloma:
        df_new = df.drop(aloma, axis = 1)
        print(df_new.head(5))
    df_new = df_new.fillna(0)
    print(df_new.head(5))
    df_newc = df_new.describe()
    print(df_newc)
    
    # transposing df_new after data cleaning
    df_transpose = df_new.transpose()
    print(df_transpose.head(5))
    


    
     # creating the year columns
    #df_new =df_new.drop('Unnamed: 66',axis=1)
    df_new['Years'] = df_new.iloc[:, 4:].sum(1)
    column_year = df_new['Years']
    print(df_new['Years'])
    
    # creating the country columns
    df_new['country'] = df_new['Country Name'] * 1
    columns_country = df_new['country']
    print(columns_country)

    # working with C02 data and considering the North American region
     #selecting argentina from the data
    argentina = pd.DataFrame(df_new.iloc[9,:])
    argentina =argentina.transpose()
    
    # chile DataFrame
    chile = pd.DataFrame(df_new.iloc[39,:])
    chile = chile.transpose()
    
    # uruguay DataFrame
    uruguay = pd.DataFrame(df_new.iloc[250,:])
    uruguay = uruguay.transpose()
    
    # merging the dataFrame for American region
    american_region = pd.concat([argentina, chile])
    american_region = pd.concat([american_region, uruguay])
    print(american_region)
    
    # droping some axis and slicing
    american_region =american_region.drop('Unnamed: 66', axis =1)
    american_real = american_region.drop(['Years','Indicator Code'],axis = 1)
    american_real = american_real.iloc[:,27:]
    american_real.set_index(['country']).plot(kind = 'bar')
    plt.title('CO2 emission in American continent')
    
    # working with work bank data to set up African region
    
    cameroon = pd.DataFrame(df_new.iloc[42,:])
    cameroon = cameroon.transpose()
    
    Egypt = df_new.iloc[67,:]
    Egypt = pd.DataFrame(df_new.iloc[67,:])
    Egypt = Egypt.transpose()
    
    congo = pd.DataFrame(df_new.iloc[44,:])
    congo = congo.transpose()
    
    angola = pd.DataFrame(df_new.iloc[4,:])
    angola =  angola.transpose()
    
    ivcoast = pd.DataFrame(df_new.iloc[41,:])
    ivcoast = ivcoast.transpose()

    african_region = pd.concat([cameroon, Egypt])
    african_region = pd.concat([african_region,congo])
    african_region = pd.concat([african_region, angola])
    african_region = pd.concat([african_region, ivcoast])
    print(african_region)
    
    
    
    
    
    # droping some axis and plotting with pandas
    african_region =african_region.drop(['Unnamed: 66','Years','Indicator Code'], axis =1)
    african_region = african_region.iloc[:,27:]
    african_region.set_index(['country']).plot(kind = 'bar')
    plt.title('CO2 emission in African ')
    
     # working with world bank data to set up European region
   
    spain = df_new.iloc[70,:]
    spain = pd.DataFrame(spain)
    spain = spain.transpose()
    
    switzerland = df_new.iloc[37,:]
    switzerland = pd.DataFrame(switzerland)
    switzerland = switzerland.transpose()
    
    cyprus = df_new.iloc[53,:]
    cyprus = pd.DataFrame(cyprus)
    cyprus = cyprus.transpose()
    
    germany = df_new.iloc[55,:]
    germany = pd.DataFrame(germany)
    germany = germany.transpose()
    
    
   
    europe_region = pd.concat([switzerland, cyprus])
    europe_region = pd.concat([europe_region,spain])
    europe_region = pd.concat([europe_region, germany])
    print(europe_region)
    
    # Removing axis and plotting europe graph
    europe_region =europe_region.drop(['Unnamed: 66','Years','Indicator Code'], axis =1)
    europe_region = europe_region.iloc[:,27:]
    europe_region.set_index(['country']).plot(kind = 'bar')
    plt.title('CO2 emission in Europe')
    print(europe_region)
    
     # working with world bank data to set up Asia region
   
    japan = df_new.iloc[119,:]
    japan = pd.DataFrame(japan)
    japan = japan.transpose()
    
    china = df_new.iloc[40,:]
    china = pd.DataFrame(china)
    china = china.transpose()
    
    UAE = df_new.iloc[8,:]
    UAE = pd.DataFrame(UAE)
    UAE = UAE.transpose()
    
    asia_region = pd.concat([japan, china])
    asia_region = pd.concat([asia_region,UAE])
    print(asia_region)
    
   
    
   # droping axes and plotting with pandas 
    asia_region =asia_region.drop(['Unnamed: 66','Years','Indicator Code'], axis =1)
    asia_region = asia_region.iloc[:,27:]
    asia_region.set_index(['country']).plot(kind = 'bar')
    plt.title('CO2 emission in Asia continent')
    
    



    
    
    


  

  
     
    return column_year, columns_country
value, df_trans = warming(df)
#print(value, df_trans)



 
#electricity = pd.read_csv('API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_5348553.csv', skiprows = 4)
#print(electricity.head())

def electricity_access():
    
    electricity = pd.read_csv('API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_5348553.csv', skiprows = 4)
    print(electricity.head())
    
    alom = ['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969',
         '1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','1990','2020','2021', 'Unnamed: 66']
    for i in alom:
        df_electricity = electricity.drop(alom, axis = 1)
    
    #electricity = pd.DataFrame(electricity)
    df_electricity =df_electricity.fillna(0)
    df_electricity = df_electricity.describe()
    print(df_electricity)
    
    # selecting african region and creating dataframe
    
    African = [['Angola',0,0,0,0,0,0,0,0,0,0,24.21274,20,26.35212,27.41278,28.47055,29.52779,30.58689,37.5,38.49,33.80219,34.89584,34.6,37.13132,38.27803,32,42,41.8132,43.01326,45.29,45.6428,46.89061],
           ['Ivory Coast',0, 0,0,0,0,0,0,0,36.5, 43.6315, 44.6481,44.64821,45.6617,46.67128,48.2,48.6847,49.65586,51.4,51.56823,52.51892,58.9,60.17105,55.37762, 60.3,57.30137,55.8,61.36259,61.9,62.6,67.14739,69.67912],
           ['Cameroon',0,0,0,0,0,0, 29,32.69753,33.84919,34.995,36.1477,38.43547,40.7,40.70678,41,46.2,44.0404,45.12301,47.1,47.28193,49,48.2,50.5914,52.7596,53.7,55.20781,59.80875,61.03086,62.2,63.48988,64.72137],
           ['Congo',0,0,0,0,0,0,0, 29.09, 30.1,31.02,31.99,32.95,33.8,34.86475,34.867,35.83,36.79,37.1,39.79,40.72,41.6,42.54,43.43,44.40,45.80,83,45.5083,46.4168,48.4075,49.51710,0],
           ['Egypt',0,0,0,93.4,95.4,95.6,95.5,96.15,96.402,96.646,96.88573,97.7,27.344,97.53618,98.8,97.90703,99.4,99.04,98.47,99.8,98.87,99.39,99.45216,99.7,99.85039,99.8,99.3,100,100,100,100]]
    for one in African:
        print(one)
    df_African= pd.DataFrame(data=African,
    columns=('Country','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'))
    
    df_African['Year'] = df_African.sum(axis =1)

    df_African['Year_percent'] = (df_African['Year']/3100)*100
    print(df_African)
    
    # plotting graph of african access to electricity
    
    plt.figure()
    plt.plot(df_African['Country'], df_African["Year_percent"], label = 'continent')
    plt.scatter(df_African['Country'], df_African["Year_percent"], label = 'continent')

    plt.xlabel("country")
    plt.ylabel("electricity")
    plt.title(" Access tto electricity in African")
    plt.legend()
    plt.show()
    
    # selecting Asia region and creating data frame
    Asia = [['UAE',100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],
           ['Japan',100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],
           ['China',0,0,0,0,0,0,0,0,0,0,97.0215,97.27279,97.5162,97.75438,97.98967,98.72, 98.46105,98.7022,98.95058,99.2089,99.7,99.85,99.96,99.99,99.996,100,100,100,100,100,100],
           ['India',0,0,0,50.9,49.8,51.41,53,54.59,56.18,60.1,58.77,55.8,62.3,64.04,64.4,65.57981,67.9,71.11,72.89,75,76.3,67.6,79.9,81.9,83.67,88,89.21,92.124,95.7,97.30527,99],
           ['Korea',99.88,99.91267,99.9429,99.96944,99.98,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,]]
    for one in Asia:
        print(one)
        
    df_Asia= pd.DataFrame(data=Asia,
    columns=('Country','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'))
    print(df_Asia)

    df_Asia['Year'] = df_Asia.sum(axis =1)

    df_Asia['Year_percent'] = (df_Asia['Year']/3100)*100
    
    mean_Asia = df_Asia['Year_percent'].mean()
    print(df_Asia)
    
    # plotting graph of Asia
    plt.figure()
    plt.plot(df_Asia['Country'], df_Asia["Year_percent"], label = 'Asia continent')
    plt.scatter(df_Asia['Country'], df_Asia["Year_percent"], label = 'Asia continent')
    plt.xlabel("country")
    plt.ylabel("electricity")
    plt.title(" Access tto electricity in Asia")
    plt.legend()
    plt.show()
    
    
    # working with Asia data and creating dataframe
    Europe = [['Switzerland',100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],
           ['Cyprus',100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],
           ['Germany',100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100],
           ['spain',100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100,100]]
    for one in Europe:
        print(one)
    df_Europe= pd.DataFrame(data=Europe,
    columns=('Country','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'))
    df_Europe['Year'] = df_Europe.sum(axis =1)

    df_Europe['Year_percent'] = (df_Europe['Year']/3100)*100
    mean_Europe = df_Europe['Year_percent'].mean()
    print(df_Europe)
    
    # plotting graph of Europe access to electricity
    
    plt.figure()
    plt.plot(df_Europe['Country'], df_Europe["Year_percent"], label = 'Europe continent')
    plt.scatter(df_Europe['Country'], df_Europe["Year_percent"], label = 'Europe continent')
    plt.xlabel("country")
    plt.ylabel("electricity")
    plt.title(" Access to electricity in Europe")
    plt.legend()
    plt.show()
    
    # working with American Continent region
    American = [['Argentina', 92.1548,92.49219,92.4982945,93.16605,93.501,93.834,94.49201,93.83447,94.16494,94.815,95.13316,95.78329,95.51106,96.22887,96.44263,96.6535,96.86385,97.07,97.292,97.52,97.75,98.82,99.1,99.22,99.342,100,99.6254,99.85,100,99.9896,100],
           ['Chile', 92.25743,94.8,94.56,95.52,95.76,95.62,96.24,97.24,96.695,97.94,97.1158,97.2837,98.78233,97.765,99.368,98.092,98.26487,99.594,99.54,99.5882,100,99.6,100,99.7,100,99.7,100,100,100,100,100],
           ['Uruguay',0,0,96.18,96.363,96.54,96.7198,95.9,97.06565,97.2325,97.395,97.6768,97.67686,97.7854,97.88633,97.98204,98.074,98.167,98.5054,98.692,98.784,99.1,99.17,99.6,99.61181,99.6118,99.657,99.7,99.8,99.9, 99.89,100],
           ['Peru',0,0,70.1,64.75,66.05,67.354,67.355,67,69.16024,72.0726,74.07206,74.079,72.4965,72.11311,74.3827,74.381,75.692,77.174,80.16,81.988,84.678,84.68,86.4232,88.12306,89.767,92.13,92.91,93.8522,94.2,94.8,99.31181],
           ['Paraguay',0,0,0,0,0,77.45,86.26,86.41,87.76,88.518,88.52,89.42,91.042,91.666,92.5584,93.249,94.687,96.75,96.45,96.68,96.891,97.431,98.24,97.84,99.02,99,99.33,98.4,99.3,99.6,100]]

    df_American= pd.DataFrame(data=American,
    columns=('Country','1990','1991','1992','1993','1994','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020'))
    df_American['Year'] = df_American.sum(axis =1)
    df_American['Year_percent'] = (df_American['Year']/3100)*100
    mean_A = df_American['Year_percent'].mean()
    print(df_American)
    
    # plotting graph of Amerigion region access to electricity
    plt.figure()
    plt.plot(df_American['Country'], df_American["Year_percent"], label = 'American continent')
    plt.scatter(df_American['Country'], df_American["Year_percent"], label = 'American continent')
    plt.xlabel("country")
    plt.ylabel("electricity")
    plt.title(" Access to electricity in American Continent")
    plt.legend()
    plt.show()
    
    
    
    # suming and plotting graph of all region together
    continent = [['African', 43.66515729032259],['Asia', 86.22746683870969 ], ['Europe',100],
            ['American', 87.92624977096776]]
    
    df_continent= pd.DataFrame(data=continent, columns=('continent','electricity_continent'))
    print(df_continent)
    
    plt.plot(df_continent['continent'], df_continent['electricity_continent'], label = 'continent')
    plt.scatter(df_continent['continent'], df_continent['electricity_continent'], label = 'continent')
    plt.xlabel('continent')
    plt.ylabel('access to electricity per 100%')
    plt.title('electricity access in all continent')
    plt.legend()
    plt.show()
    


electricity_access()

# this function calculate the correlation of all indicators in American and European Continent
def all_indicators():
    all_indicator = pd.read_csv('API_19_DS2_en_csv_v2_5346672.csv', skiprows = (3))
    print(all_indicator)
    print(all_indicator.columns)
    
    df = all_indicator[['Indicator Name','2000', '2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
       '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',]]
    
    df_argentina = all_indicator.iloc[684:759,:]
    df_argentina = df_argentina.fillna(0)
    print(df_argentina)
    
    All_indicator = [['2015',185550.0031,99.6253,5.156686,10.63219,13.40601,42.78468,4779.1,49.48226,26.24158],
           ['2014',179600,99.71,6.712704,10.71382,13.89989,43.8533,4827.3,48.199968,29.60794],
           ['2013',183250, 99.42, 6.052918, 10.79545, 14.39377, 44.92193,4795.6,47.23536,29.36913],
           ['2012',177960,99.62897,5.781744,10.87708,14.4784,45.5813,4162.4,48.31413,27.23204],
           ['2011',1776640,99.6058,6.998734,10.95871,14.11431,45.79196,4596.3,44.8155,30.64258],
            ['2010',167220,99.8,7.132167,11.04034,13.88652,46.1391,4876.2,49.98204,26.81319]]
    for one in All_indicator:
        print(one)
    df_indicators= pd.DataFrame(data=All_indicator,
    columns=('Year','CO2 emissions', 'Access to electricity','Agriculture','Forest area','Arable land','Agricultural land','Cereal yield','natural gas sources','hydroelectricity sources'))
    print(df_indicators)
    df = df_indicators[['Year', 'CO2 emissions', 'Access to electricity',
       'Agriculture','Forest area', 'Arable land',
       'Agricultural land', 'Cereal yield','natural gas sources',
       'hydroelectricity sources']]
    correl = df_indicators.corr()
    print(correl)
    # creating the heatmap to view the correllation
    plt.figure(figsize = (10,5))
    sns.heatmap(data = correl, annot = True)
    
    
    
     # working with europe data 
    df_europe = [['2015', 551770,3.308467384, 69.15055862, 14.92576527,46.37463487,8817.6,7.291326144,114.6486124],['2014', 567919.9753, 3.345600766, 69.24349106, 14.86105276,46.38451389, 8716.6,6.986281585,112.1036975],
                 ['2013', 589379.9782,3.467464481, 69.33642847, 14.78874796, 46.28583946,7461.9, 7.020, 110.4785616],
                 ['2012', 581699,3.645931016, 69.42937085, 14.74174501, 46.19475195, 9491.3, 7.205739953, 109.6613326],
                 ['2011',594880.0278,3.385788279,69.52309649,14.6975321,46.0995923,9442.3,7.907163138,112.3594044],
                 ['2010', 586010.0288,2.964376255,69.61605133,14.68089466,45.90003878,9380.9,8.57408882,107.5661802]]
    
   
    df_europe= pd.DataFrame(data=df_europe,
    columns=('Year','CO2 emissions', 'Agriculture','Forest area','Arable land','Agricultural land','Cereal yield','natural gas sources ','hydroelectric sources'))
    print(df_europe.columns)
    print(df_europe)
    
    # creating correlation and heat map
    df_europe = df_europe[['Year', 'CO2 emissions', 'Agriculture', 'Forest area', 'Arable land',
       'Agricultural land', 'Cereal yield', 'natural gas sources ',
       'hydroelectric sources']]
    correl_europe = df_europe.corr()
    print("this is the correlation" + str(correl_europe))
    
    # heatmap of europe
    plt.figure(figsize = (10,5))
    sns.heatmap(data = correl_europe, annot = True)
    
    
    
    # calculate skewness for all indicators in american continent 
    skewness_american =df_indicators.iloc[:,1:]
    for col in skewness_american:
        print(col)
        print(skew(skewness_american[col]))
        plt.figure()
        sns.distplot(skewness_american[col])
        plt.show()
        
    # calculate skewness for all indicators in european continent    
    skewness_europe = df_europe.iloc[:,1:]
    for col in skewness_europe:
        print(col)
        print(skew(skewness_europe[col]))
        plt.figure()
        sns.distplot(skewness_europe[col])
        plt.show()
    
    
    
      
    
all_indicators()



