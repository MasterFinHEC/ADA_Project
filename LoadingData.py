import numpy as np 
import pandas as pd 
import world_bank_data as wb
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy import distance

class data:

	def __init__(self):
		self.y = None
		self.X = None

	def loadData(self):

		print('Loading John-Hopkins covid 19 Data')
		# Loading Covid 19 Data
		public_url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
		corona_data = pd.read_csv(public_url)
		corona_data.drop(['Lat','Long','Province/State'],axis=1)
		countries = sorted(corona_data['Country/Region'].unique())
		country_data = corona_data.groupby('Country/Region').sum()
		country_data = country_data.drop(['Lat','Long'],axis=1)
		country_data = country_data.rename(columns={'Country/Region':'Country'},index={'US':'United States'})


		print('Loading World Bank indicators')
		# Get GDP data
		GDP = pd.DataFrame(wb.get_series('NY.GDP.MKTP.CD',mrv=1))
		GDP = GDP.droplevel(level=[1,2]) # Droping multi level indexing

		# Get gini Index
		Gini = pd.DataFrame(wb.get_series('SI.POV.GINI',date = '2010'))
		Gini = Gini.droplevel(level=[1,2]) # Droping multi level indexing

		# Get population data
		Pop = pd.DataFrame(wb.get_series('SP.POP.TOTL',mrv=1))
		Pop = Pop.droplevel(level=[1,2]) # Droping multi level indexing

		# Get Health System Data
		Health = pd.DataFrame(wb.get_series('SH.MED.BEDS.ZS',date = '2010'))
		Health = Health.droplevel(level=[1,2]) # Droping multi level indexing

		# Get Density Data
		Dens = pd.DataFrame(wb.get_series('EN.POP.DNST',mrv=1))
		Dens = Dens.droplevel(level=[1,2])

		# Get Trade data
		Trade = pd.DataFrame(wb.get_series('NE.TRD.GNFS.ZS',mrv=1))
		Trade = Trade.droplevel(level=[1,2])

		# Get Child mortality data
		Child = pd.DataFrame(wb.get_series('SP.DYN.IMRT.IN',mrv=1))
		Child = Child.droplevel(level=[1,2])
		Child = Child/1000

		
		print('Loading data from Stansford University')
		politics = pd.read_csv('politics.csv')
		politics = politics.set_index('Country Name')
		politics = politics.drop(['Series Name','Country Code','Series Code'], axis=1)
		politics = politics.rename(columns = {'2018 [YR2018]':'Political Stability'})

		GOV = pd.read_csv('governement.csv')
		GOV = GOV.set_index('Country Name')
		GOV = GOV.drop(['Series Name','Country Code','Series Code'], axis=1)
		GOV = GOV.rename(columns = {'2018 [YR2018]':'GOV'})

		print('Loading the Economist Data')
		# Economist businnes unit
		df = pd.read_excel('DemocracyIndex.xlsx')
		year = df['time'] == 2018
		DEM = df[year]
		DEM = DEM.drop(['geo','a','b','c','d','e','time','f'],axis = 1)
		DEM = DEM.set_index('name')
		DEM = DEM.rename(columns={'name':'Country'})

		# Continent data
		Cont = pd.read_csv('Countries-Continents.csv')
		Cont = Cont.set_index('Country')
		Cont = Cont.rename(index={'US':'United States'})

		print('Merging all data and selecting only the countries with all the data available')
		allData = country_data.join([GDP,Gini,DEM,Pop,Health,Child,Dens,Trade,Cont,politics,GOV])
		allData.rename(columns={'NY.GDP.MKTP.CD':'GDP',
                          'SI.POV.GINI':'Gini',
                          'Democracy index (EIU)':'Dem',
                       'SP.POP.TOTL':'Pop',
                       'SH.MED.BEDS.ZS': 'Health',
                        'SP.DYN.IMRT.IN':'Child',
                       'EN.POP.DNST':'Dens',
                       'NE.TRD.GNFS.ZS':'Trade',
                        'Political Stability	':'Political Stability'
                      },inplace=True)
		allData = allData.dropna()


		print('Computing distance between countries !')
		geolocator = Nominatim(user_agent="my-application")
		Distance = []
		count=0
		countries = list(allData.index)
		Wuhan = geolocator.geocode("Wuhan")
		Wuhan = (Wuhan.latitude, Wuhan.longitude)

		for i in countries[0:16]:
		    c = geolocator.geocode(i)
		    Distance.append(distance.distance((c.latitude, c.longitude), Wuhan).km)
		    count += 1
		print('25 %')
		for i in countries[16:33]:
		    c = geolocator.geocode(i)
		    Distance.append(distance.distance((c.latitude, c.longitude), Wuhan).km)
		    count += 1
		print('50 %')
		for i in countries[33:45]:
		    c = geolocator.geocode(i)
		    Distance.append(distance.distance((c.latitude, c.longitude), Wuhan).km)
		    count += 1
		print('75 %')
		for i in countries[45:59]:
		    c = geolocator.geocode(i)
		    Distance.append(distance.distance((c.latitude, c.longitude), Wuhan).km)
		    count += 1

		Distances = pd.DataFrame(Distance, index =list(allData.index),columns =['Distance'])

		allData = allData.join([Distances])
		print('100 %')

		self.y = allData.drop(['GDP', 'Gini', 'Dem', 'Pop', 'Health','Child',
                  'Dens', 'Trade','Continent','Political Stability','GOV','Distance'],axis = 1)

		self.X = allData.loc[:,['GDP', 'Gini', 'Dem', 'Pop', 'Health','Child',
                  'Dens', 'Trade','Continent','Political Stability','GOV','Distance']]
        

