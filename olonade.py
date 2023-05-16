# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:49:26 2023

@author: Olonade
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.optimize as opt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
from sklearn.cluster import KMeans

import cluster_tools as ct
import errors as err
import importlib
importlib.reload(ct)


df_emission = pd.read_csv('API_EN.ATM.CO2E.PP.GD.KD_DS2_en_csv_v2_5362896.csv', skiprows=(4))
df_gdp = pd.read_csv('API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv', skiprows=(4))

help(KMeans)
def cluster(data):
    
    
    import sklearn.cluster as cluster
    # reading data
    df_emission = pd.read_csv('API_EN.ATM.CO2E.PP.GD.KD_DS2_en_csv_v2_5362896.csv', skiprows=(4))
    print(df_emission)
    
    df_gdp = pd.read_csv('API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv', skiprows=(4))
    print(df_gdp.describe())
    
    # cleaning the data and transposing
    df_emission = df_emission.drop(['1960','1961','1962','1963','1964','1965','1966','1967','1968','1969','1970','1971','1972','1973','1974','1975','1976','1977','1978','1979','1980','1981','1982','1983','1984','1985','1986','1987','1988','1989','Unnamed: 66'], axis=1)
    
    # transposing
    df = df_emission.transpose()
    
    #using logical slicing
    df_emission = df_emission[['Country Name','1990','2000','2010','2019']]
    df_emission = df_emission.dropna()
    print(df_emission)
    

    # transposing gdp data
    df2 = df_gdp.transpose()
    # working gdp data, cleaning and slicing
    df_gdp = df_gdp[['Country Name', '1990', '2000','2010', '2019']]
    df_gdp = df_gdp.dropna()
    print(df_gdp)
    
    # making a copy, year 1990
    df_emission1990 = df_emission[["Country Name", "1990"]].copy()
    df_gdp1990 = df_gdp[["Country Name", "1990"]].copy()
    
    #making a copy, year 2000
    df_emission2000 = df_emission[["Country Name", "2000"]].copy()
    df_gdp2000 = df_gdp[["Country Name", "2000"]].copy()
   
    #making a copy ,year 2010
    df_emission2010 = df_emission[["Country Name", "2010"]].copy()
    df_gdp2010 = df_gdp[["Country Name", "2010"]].copy()
    print(df_gdp2010)
    
    # working wth data 1990
    print(df_emission1990.describe())
    print(df_gdp1990.describe())
    
    #merging dataframe and dropna
    df_1990 = pd.merge(df_emission1990, df_gdp1990, on="Country Name", how="outer")
    print(df_1990.describe())
    df_1990.to_excel("emi_gdp1990.xlsx")
    df_1990 = df_1990.dropna()
    df_1990.describe()
    
    # renaming columns
    df_1990 = df_1990.rename(columns={"1990_x":"emission", "1990_y":"gdp"})
    print(df_1990)
    
    # using scatter matrix 
    pd.plotting.scatter_matrix(df_1990, figsize=(12, 12), s=5, alpha=0.8)
    
    # correlation of variables
    df1990 = df_1990.corr()
    print(df1990)
    
    #clustering
    
    df_em1990 = df_1990[["emission", "gdp"]].copy()

    # normalise
    df_norm, df_min, df_max = ct.scaler(df_em1990)
    
    print("n    score")
    # loop over number of clusters
    
        
        # loop over number of clusters
    for ncluster in range(2, 10):
        
        
            # set up the clusterer with the number of expected clusters
            kmeans = cluster.KMeans(n_clusters=ncluster)

            # Fit the data, results are stored in the kmeans object
            kmeans.fit(df_em1990)     # fit done on x,y pairs

            labels = kmeans.labels_
    
            # extract the estimated cluster centres
            cen = kmeans.cluster_centers_

            # calculate the silhoutte score
            print(ncluster, skmet.silhouette_score(df_em1990, labels))
            
            
    ncluster = 4

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_em1990)     # fit done on x,y pairs

    labels = kmeans.labels_
    
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    xcen = cen[:, 0]
    ycen = cen[:, 1]


    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_em1990["emission"], df_em1990["gdp"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("emission")
    plt.ylabel("gdp")
    plt.title('world cluster 1990')
    plt.savefig('cluster41990.png', dpi = 300)
    
    plt.show()   
    
    # trying cluster = 5
    
    ncluster = 5

    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)

    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_em1990)     # fit done on x,y pairs

    labels = kmeans.labels_
    
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_

    xcen = cen[:, 0]
    ycen = cen[:, 1]


    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_em1990["emission"], df_em1990["gdp"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("emission")
    plt.ylabel("gdp")
    plt.title('world cluster for the year 1990')
    plt.savefig('cluster51990.png', dpi = 300)
    plt.show()
    
    
    # getting the original clustter
    cen = ct.backscale(cen, df_min, df_max)
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_1990["emission"], df_1990["gdp"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("emission")
    plt.ylabel("gdp")
    plt.title('world cluster original scale (1990) ')
    plt.savefig('cluster_orig1990.png', dpi = 300)
    plt.show()
    
    
    # working with cluster for all countries for the year 2000
    
    df_2000 = pd.merge(df_emission2000, df_gdp2000, on="Country Name", how="outer")
    print(df_2000.describe())

    df_2000.to_excel("emi_gdp2000.xlsx")
    df_2000 = df_2000.dropna()
   
    # rename column
    df_2000 = df_2000.rename(columns={"2000_x":"emission", "2000_y":"gdp"})
    print(df_2000.corr())
    
    # getting the score of the cluster
    df_emi2000 = df_2000[["emission", "gdp"]].copy()

    # normalise
    df_norm, df_min, df_max = ct.scaler(df_emi2000)

    
    # loop over number of clusters
    for ncluster in range(2, 10):
    
        
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(df_emi2000)      
        labels = kmeans.labels_
        cen = kmeans.cluster_centers_
        print(ncluster, skmet.silhouette_score(df_emi2000, labels))

    # displaying the cluster
        ncluster = 6
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(df_emi2000)     
        labels = kmeans.labels_
        cen = kmeans.cluster_centers_
        xcen = cen[:, 0]
        ycen = cen[:, 1]

        plt.figure(figsize=(8.0, 8.0))
        cm = plt.cm.get_cmap('tab10')
        plt.scatter(df_2000["emission"], df_2000["gdp"], 10, labels, marker="o", cmap=cm)
        plt.scatter(xcen, ycen, 45, "k", marker="d")
        plt.xlabel("emission")
        plt.ylabel("gdp")
        plt.title('world cluster for the year 2000')
        plt.savefig('cluster2000.png', dpi = 300)
    
        plt.show()
        
        
        # move the cluster centres to the original scale
        cen = ct.backscale(cen, df_min, df_max)
        xcen = cen[:, 0]
        ycen = cen[:, 1]
        plt.figure(figsize=(8.0, 8.0))
        cm = plt.cm.get_cmap('tab10')
        plt.scatter(df_2000["emission"], df_2000["gdp"], 10, labels, marker="o", cmap=cm)
        plt.scatter(xcen, ycen, 45, "k", marker="d")
        plt.xlabel("emission")
        plt.ylabel("gdp")
        plt.title('world cluster original scale(2000)')
        plt.savefig('cluster_orig2000.png', dpi = 300)
        plt.show()
        
        
        # working with 2010 dataset
        print(df_emission2010.describe())
        print(df_gdp2010.describe())
        
        df_2010 = pd.merge(df_emission2010, df_gdp2010, on="Country Name", how="outer")
        
        df_2010 = df_2010.dropna() 
        print(df_2010.describe())
        df_2010 = df_2010.rename(columns={"2010_x":"emission", "2010_y":"gdp"})
        print(df_2010.corr())
        
        df_emi2010 = df_2010[["emission", "gdp"]].copy()

        # normalise
        df_cluster, df_min, df_max = ct.scaler(df_emi2010)
        
        # loop over number of clusters
        for ncluster in range(2, 10):
            
            
            # set up the clusterer with the number of expected clusters
            kmeans = cluster.KMeans(n_clusters=ncluster)
            kmeans.fit(df_emi2010)     
            labels = kmeans.labels_
            cen = kmeans.cluster_centers_
            print(ncluster, skmet.silhouette_score(df_emi2010, labels))
        
        
        ncluster = 7

        # visualising the cluster with the number of expected clusters
        kmeans = cluster.KMeans(n_clusters=ncluster)

       
        kmeans.fit(df_cluster)      
        labels = kmeans.labels_
    
        
        cen = kmeans.cluster_centers_

        xcen = cen[:, 0]
        ycen = cen[:, 1]



        plt.figure(figsize=(8.0, 8.0))

        cm = plt.cm.get_cmap('tab10')
        plt.scatter(df_emi2010["emission"], df_emi2010["gdp"], 10, labels, marker="o", cmap=cm)
        plt.scatter(xcen, ycen, 45, "k", marker="d")
        plt.xlabel("emission")
        plt.ylabel("gdp")
        plt.title('world cluster for the year 2010')
        plt.savefig('cluster2010.png', dpi = 300)
    
        plt.show()
        
        
        # move the cluster centres to the original scale
        cen = ct.backscale(cen, df_min, df_max)
        xcen = cen[:, 0]
        ycen = cen[:, 1]

       
        plt.figure(figsize=(8.0, 8.0))

        cm = plt.cm.get_cmap('tab10')
        plt.scatter(df_2010["emission"], df_2010["gdp"], 10, labels, marker="o", cmap=cm)
        plt.scatter(xcen, ycen, 45, "k", marker="d")
        plt.xlabel("emission")
        plt.ylabel("gdp")
        plt.title('world cluster centered to original scale (2010)')
        plt.savefig('cluster_orig2010.png', dpi=300)
        plt.show()
        


        return df_emission

last = cluster(df_emission)
    
    
def fitting(df_gdp):
    
    
    df_gdp = pd.read_csv('API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv', skiprows=(4))
    
    # using logical slicing
    gdp_usa = df_gdp.iloc[251:252,:]
    print(gdp_usa)
    
    # cleaning the data
    gdp_usa = gdp_usa.dropna(axis=1)
    
    # setting up new dataframe
    aloma = [[1961, 0.861821], [1962, 4.480669], [1963, 4.480669], [1964, 4.340549],[1965, 5.070809], [1966, 5.277114], [1967, 1.389951], [1968, 3.758819], [1969, 2.09737], [1970, -1.438451], [1971, 1.995600], [1972, 4.138097], [1973, 4.642156], [1974, -1.445134], [1975, -1.184581], [1976, 4.391463], [1977, 3.577147], [1978, 4.422985], [1979, 2.033887], [1980, -1.209298], [1981, 1.5363202], [1982, -2.73457], [1983, 3.631979], [1984, 6.312168], [1985, 3.250656], [1986, 2.510886], [1987, 2.538624], [1988, 3.235416], [1989, 2.698167], [1990, 0.741486], [1991, -1.434200], [1992, 2.096613], [1993, 1.405709], [1994, 2.760882], [1995,1.468718], [1996, 2.572259], [1997, 3.197212], [1998, 3.270511], [1999, 3.597985], [2000, 2.925441], [2001,-0.039910], [2002, 0.756774], [2003, 1.916481], [2004, 2.895848], [2005, 2.533784], [2006, 1.796486], [2007, 1.044930], [2008, -0.820368], [2009, -3.450016], [2010, 1.860292], [2011, 0.814519],[2012, 1.533102], [2013, 1.138692], [2014, 1.540381], [2015, 1.953004], [2016, 0.933375], [2017, 1.597136], [2018, 2.404868], [2019, 1.829668], [2020, -3.697922], [2021, 5.82014]]
    df_usa = pd.DataFrame(data = aloma, columns = ['year', 'GDP'])
    print(df_usa) 
    
    # define the exponential function to check if it is a good fit
    def exponential(t, n0, g):
        
        
        t = t - 2010
        f = n0 * np.exp(g*t)
        return f
    
    
    print(type(df_usa["year"].iloc[1]))
    df_usa["year"] = pd.to_numeric(df_usa["year"])
    print(type(df_usa["year"].iloc[1]))
    
    #ploting to visualize the graph
    df_usa.plot('year','GDP',kind = 'scatter')
    plt.xlabel('year')
    plt.ylabel('GDP')
    plt.title('USA GDP')
    
    # checking for good fit
    param, covar = opt.curve_fit(exponential, df_usa["year"], df_usa["GDP"])
    print(param)
    plt.figure()
    plt.plot(df_usa["year"], exponential(df_usa["year"], 1.36297, -0.02), label = "trial fit")
    plt.plot(df_usa["year"], df_usa["GDP"], label = 'true data')
    plt.xlabel("year")
    plt.ylabel('GDP')
    plt.title('Graph of USA GDP against year')
    plt.legend()
    plt.savefig('fittingexp.png', dpi = 300)
    plt.show()
    
    
    # further test
    df_usa["fit"] = exponential(df_usa["year"], *param)
    df_usa.plot("year", ["GDP", "fit"])
    plt.xlabel('year')
    plt.ylabel('GDP')
    plt.title('Graph of USA GDP against year')
    plt.savefig('final_fittingexp.png', dpi = 300)
    plt.show()
    
    
    # exponential function do not fit well, we use polynomial to solve this problem
    def poly(x, a, b, c, d, e):
        
        
        x = x - 2010
        f = a + b*x + c*x**2 + d*x**3 + e*x**4
        return f
    
    
    param, covar = opt.curve_fit(poly, df_usa["year"], df_usa["GDP"])
    print(param)
    df_usa["fit"] = poly(df_usa["year"], *param)
    df_usa.plot("year", ["GDP", "fit"])
    plt.xlabel('year')
    plt.ylabel('GDP')
    plt.title('Graph of USA GDP against year')
    plt.savefig('fittingpoly.png', dpi = 300)
    plt.show()
    
    # forecasting the GDP in 2031
    year = np.arange(1960, 2031)
    forecast = poly(year, *param)
    plt.figure()
    plt.plot(df_usa["year"], df_usa["GDP"], label="GDP")
    plt.plot(year, forecast, label="forecast")
    plt.xlabel("year")
    plt.ylabel("GDP")
    plt.title('USA GDP for 2030 forecasted')
    plt.legend()
    plt.savefig('poly_forecast.png', dpi = 300)
    plt.show()
    
    # estimating the errors and calculating sigma
    sigma = np.sqrt(np.diag(covar))
    df_usa["fit"] = poly(df_usa["year"], *param)
    df_usa.plot("year", ["GDP", "fit"])
    plt.show()

    print("turning point", param[2], "+/-", sigma[2])
    print("GDP at turning point", param[0]/100, "+/-", sigma[0]/100)
    print("growth rate", param[1], "+/-", sigma[1])
    
    # error bandgap
    sigma = np.sqrt(np.diag(covar))
    low, up = err.err_ranges(year, poly, param, sigma)
    plt.figure()
    plt.plot(df_usa["year"], df_usa["GDP"], label="GDP")
    plt.plot(year, forecast, label="forecast")
    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("GDP")
    plt.legend()
    plt.savefig('errorband.png', dpi=300)
    plt.show()
    print(param)
    
    return df_gdp

curve = fitting(df_gdp)
    

# we investigated USA GDP cluster with respect to the world
    
def usa():
    
    
    # logical slicing
    import sklearn.cluster as cluster
    aloma = [[1961, 0.861821], [1962, 4.480669], [1963, 4.480669], [1964, 4.340549],[1965, 5.070809], [1966, 5.277114], [1967, 1.389951], [1968, 3.758819], [1969, 2.09737], [1970, -1.438451], [1971, 1.995600], [1972, 4.138097], [1973, 4.642156], [1974, -1.445134], [1975, -1.184581], [1976, 4.391463], [1977, 3.577147], [1978, 4.422985], [1979, 2.033887], [1980, -1.209298], [1981, 1.5363202], [1982, -2.73457], [1983, 3.631979], [1984, 6.312168], [1985, 3.250656], [1986, 2.510886], [1987, 2.538624], [1988, 3.235416], [1989, 2.698167], [1990, 0.741486], [1991, -1.434200], [1992, 2.096613], [1993, 1.405709], [1994, 2.760882], [1995,1.468718], [1996, 2.572259], [1997, 3.197212], [1998, 3.270511], [1999, 3.597985], [2000, 2.925441], [2001,-0.039910], [2002, 0.756774], [2003, 1.916481], [2004, 2.895848], [2005, 2.533784], [2006, 1.796486], [2007, 1.044930], [2008, -0.820368], [2009, -3.450016], [2010, 1.860292], [2011, 0.814519],[2012, 1.533102], [2013, 1.138692], [2014, 1.540381], [2015, 1.953004], [2016, 0.933375], [2017, 1.597136], [2018, 2.404868], [2019, 1.829668], [2020, -3.697922], [2021, 5.82014]]
    df_usa = pd.DataFrame(data = aloma, columns = ['year', 'GDP'])
    print(df_usa) 
    df_usa_gdp = df_usa[['GDP']]
    print(df_usa_gdp)
    df_usa_gdp['emission'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.030473019, 0.023921592, 0.024094918, 0.023579321, 0.022315381, 0.022546365, 0.069878181, 0.084483919, 0.079298914, 0.083365696, 0.083454241, 0.097083249, 0.116236159, 0.12576826, 0.130798613, 0.140727216, 0.161998587, 0.176194281, 0.16880937, 0.18139915, 0.194121899,0.181508098,0.165355342,0.162600176, 0.169540588, 0.176849315, 0.209336309,0.201921847, 0.204398275, 0.188173258]
    print(df_usa_gdp)
    print(df_usa_gdp.describe())
    print(df_usa_gdp.corr)
    
    # clustering operation
    df_usa_cluster = df_usa_gdp[["GDP", "emission"]].copy()
    df_cluster, df_min, df_max = ct.scaler(df_usa_cluster)
    
    
    for ncluster in range(2, 10):
        
        
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(df_usa_cluster)     
        labels = kmeans.labels_
        cen = kmeans.cluster_centers_
        print(ncluster, skmet.silhouette_score(df_usa_cluster, labels))
        df_cluster, df_min, df_max = ct.scaler(df_usa_cluster)
        
        
    
    ncluster = 8
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_usa_cluster)     
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    xcen = cen[:, 0]
    ycen = cen[:, 1]

    plt.figure(figsize=(8.0, 8.0))

    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_usa_cluster["GDP"], df_usa_cluster["emission"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("GDP")
    plt.ylabel("emission")
    plt.title('USA cluster for 61 years')
    plt.savefig('usa_cluster.png', dpi=300)
    plt.show()
    
    
    # cluster centered to original scale
    cen = ct.backscale(cen, df_min, df_max)
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_usa_gdp["GDP"], df_usa_gdp["emission"], 10, labels, marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("GDP")
    plt.ylabel("emission")
    plt.title('USA cluster centered to original scale')
    plt.savefig('usa_cluster_orig.png', dpi=300)
    plt.show()



usa()
