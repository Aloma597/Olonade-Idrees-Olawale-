# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:17:57 2023

@author: Olonade
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
import numpy as np
df_covid_vaccination = pd.read_csv('vaccination-data.csv')
print(df_covid_vaccination)

# declaration of variable 
i = df_covid_vaccination['TOTAL_VACCINATIONS']
j = df_covid_vaccination['PERSONS_FULLY_VACCINATED']
k = df_covid_vaccination['PERSONS_VACCINATED_1PLUS_DOSE']
l = df_covid_vaccination['PERSONS_VACCINATED_1PLUS_DOSE_PER100']
m = df_covid_vaccination ['PERSONS_FULLY_VACCINATED_PER100']
n = df_covid_vaccination['PERSONS_BOOSTER_ADD_DOSE']
q = df_covid_vaccination["WHO_REGION"]
 
AFRO = 'WHO Regional Office for Africa' 
AMRO = 'WHO Regional Office for the Americas'
SEARO =' WHO Regional Office for South-East ASIA'
EURO = 'WHO Regional Office for Europe'
EMRO = 'WHO Regional Office for Mediterranean'
WPRO = 'WHO Regional Office for Western Pacific' 


# created a function read_file that takes six argument            
def read_file(i,j,k,l,m,n):
   return i,j,k,l,m,n
i,j,k,l,m,n = read_file(i,j,k,l,m,n)

# plotted the line plot
plt.figure()
plt.figure()
plt.plot(df_covid_vaccination['TOTAL_VACCINATIONS'], df_covid_vaccination['PERSONS_FULLY_VACCINATED'], 'k--', label= 'PFL')
plt.plot(df_covid_vaccination['TOTAL_VACCINATIONS'],  df_covid_vaccination['PERSONS_VACCINATED_1PLUS_DOSE'],':',
           label = 'PF1')
plt.plot(df_covid_vaccination['TOTAL_VACCINATIONS'],  df_covid_vaccination['PERSONS_VACCINATED_1PLUS_DOSE_PER100'],
         label = 'PF100')
plt.plot(df_covid_vaccination['TOTAL_VACCINATIONS'],  df_covid_vaccination['PERSONS_FULLY_VACCINATED_PER100'],  
         '-.', label = 'PFL100')
plt.title("COVID VACCINATIONS", size=10)

plt.plot(df_covid_vaccination['TOTAL_VACCINATIONS'],  df_covid_vaccination['PERSONS_BOOSTER_ADD_DOSE'],
         ':', label = 'PF100B')



plt.xlabel('Total Vaccines')
plt.xlim(min(df_covid_vaccination['TOTAL_VACCINATIONS']),
             df_covid_vaccination['TOTAL_VACCINATIONS'].max())
plt.ylabel('People vaccinated')
plt.legend(loc= 'upper left')
plt.savefig(" line plot.png")    
plt.show()


# Programs to view the number of vaccines used by people in all WHO region

df_covid_vaccination = pd.read_csv('vaccination-data.csv', index_col=0)
print(df_covid_vaccination)

def bar_file(q,j,k,l,m,n):
   return q,j,k,l,m,n
q,j,k,l,m,n = bar_file(q,j,k,l,m,n)

plt.figure()

plt.bar(df_covid_vaccination["WHO_REGION"], df_covid_vaccination["PERSONS_FULLY_VACCINATED"], width=0.8)
plt.title("full vaccination")
plt.xlabel("WHO REGION")
plt.ylabel("persons fully vaccinated")
plt.savefig(" bar chart full vaccination.png")
plt.show()

plt.figure()
plt.bar(df_covid_vaccination["WHO_REGION"], df_covid_vaccination["PERSONS_VACCINATED_1PLUS_DOSE_PER100"], width=0.8)
plt.title("Partial Vaccination")
plt.xlabel("WHO REGION")
plt.ylabel("At least one dose per 100")
plt.savefig(" bar chart part with booster.png")
plt.show()

plt.figure()
plt.bar(df_covid_vaccination["WHO_REGION"], df_covid_vaccination["PERSONS_VACCINATED_1PLUS_DOSE"], width=0.8)
plt.title("Partial Vaccination")
plt.xlabel("WHO REGION")
plt.savefig(" bar chart partial vaccination.png")
plt.ylabel("At least one dose")


plt.figure()
plt.bar(df_covid_vaccination["WHO_REGION"], df_covid_vaccination["PERSONS_BOOSTER_ADD_DOSE"], width=0.8)
plt.title("full vaccination with booster ")
plt.xlabel("WHO REGION")
plt.ylabel("persons fully vaccinated with booster")
plt.savefig("bar chart full vaccination with booster.png")
plt.show()

plt.figure()
plt.bar(df_covid_vaccination["WHO_REGION"], df_covid_vaccination["PERSONS_BOOSTER_ADD_DOSE_PER100"], width=0.8)
plt.title("full vaccination with booster per 100 ")
plt.xlabel("WHO REGION")
plt.ylabel("persons fully vaccinated with booster per 100")
plt.savefig(" bar chart full vaccination with booster per 100.png")
plt.show()


# This program used pie chart to visualize the vaccination of people in all WHO region
def pie_file(q,j,k,l,m,n):
   return q,j,k,l,m,n
q,j,k,l,m,n = bar_file(q,j,k,l,m,n)

REGION_WHO= ['EMRO','EURO','AFRO','AMRO','WPRO','SEARO', 'OTHER']


value= [348090522,561880427, 355657068, 647166980, 4502787047,1462162402, 26681]

plt.figure()
plt.pie(value, labels =REGION_WHO)
plt.title('Partial vaccination with at least one dose')
plt.savefig("pie chart partial vaccination.png")
plt.show()

Full_vaccination_value = [349360607, 492954279, 314859155, 387953011, 1598488008,
                          1302123981, 26346]

plt.figure()
plt.pie(Full_vaccination_value, labels =REGION_WHO)
plt.title('full vaccination')
plt.savefig(" pie chart full vaccination.png")
plt.show()

full_vaccination_with_booster= [121620238, 324288485, 38100550, 405287595,
                                1051547360, 382559245, 18310]

plt.figure()
plt.pie(full_vaccination_with_booster, labels = REGION_WHO)
plt.title('full vaccination with booster')
plt.savefig(" pie chart full vaccination with booster.png")

plt.show()


full_vaccination_and_boster_per100 = [542.39, 2264.343, 315.173, 1445.11,
                                       1471.634, 178.902, 47.255]

plt.figure()
plt.pie(full_vaccination_and_boster_per100, labels = REGION_WHO)
plt.title('full vaccination with booster per100')
plt.savefig(" pie chart full vaccination with booster per 100.png")
plt.show()

Data_vaccination_subplot = [value, Full_vaccination_value, 
                            full_vaccination_with_booster, 
                            full_vaccination_and_boster_per100]

plt.figure(figsize = (10,10))
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.subplot(2, 2, 1)
plt.pie(value, labels =REGION_WHO)
plt.title('Partial vaccination with at least one dose')

plt.subplot(2, 2, 2)
plt.pie(Full_vaccination_value, labels =REGION_WHO)
plt.title('full vaccination')

plt.subplot(2, 2, 3)
plt.pie(full_vaccination_with_booster, labels = REGION_WHO)
plt.title('full vaccination with booster')

plt.subplot(2, 2, 4)
plt.pie(full_vaccination_and_boster_per100, labels = REGION_WHO)
plt.title('full vaccination with booster per100')
plt.savefig("four pie chart.png")
plt.show()

Name_of_Vaccine = ['Astrazeneca', 'Beijing C', 'Janssen', 'SII-Covishield', 
                   'moder mRNa','Bharat-Covaxi']

no_of_vaccine_used_in_full_vaccination_per_100 =np.array([8760.313, 1202.78, 1035.373, 164.382, 
                                          760.193, 175.248])



# Programs to show the types of Vaccine used for vaccination
Region = [['EMRO',348090522],['EURO',561880427],['AFRO',355657068],
                  ['AMRO', 647166980], ['WPRO',4502787047],
                  ['SEARO', 1462162402], ['OTHER', 26681]]

plt.figure()
plt.pie(no_of_vaccine_used_in_full_vaccination_per_100,labels = Name_of_Vaccine)
plt.title('no_of vaccine_used_in_full_vaccination')
plt.savefig("pie chart with no of  vaccine.png")
plt.show()




 










