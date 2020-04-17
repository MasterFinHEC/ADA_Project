# ADA_Project
Repository for the Advanced Data Analytics project from the Master of science in finance from HEC Lausanne. 

![HEC Lausanne Logo](https://fr.wikipedia.org/wiki/Facult%C3%A9_des_hautes_%C3%A9tudes_commerciales_de_l'universit%C3%A9_de_Lausanne#/media/Fichier:Logo_HEC_Lausanne.png)

## Project
The project is to try and find good predictors of a pandemic reach on a country. 

Therefore using data on : 

- Economic Activity (GDP per capita)
- Level of democracy
- Density of populaiton
- Inequality (Gini Index)
- Quality of health System
- Proportion of people believing in God
- Openness Index (typically Trade/GDP data)

We want to see if there is a way to predict the way a country will be affected by a pandemic using the Covid-19 data. 

Then, the model could be applied to other pandemic to see if the relation is the same. 

### Data

#### Covid - 19 Data
Taken directly from the John Hopkins for decease center, they are updated every day when running the code.

You can get them *[here](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv).

#### World Bank Data
Using the World Bank API, we extract social, economic and demographic data. 

You juste need to install it as follow on your computer
```
pip install world_bank_data
```
The library is then loaded in the script.

You can check how this works *[here](https://github.com/mwouts/world_bank_data).

#### Downloaded Data 
We have downloaded two dataset that we couldn't have through an API, there are in the repository. 

1. Democracy Index . From The Economist.
2. A Data set mapping the countries to the continents. 


## Authors

* **Alessia Di Pietro** 
* **Antoine-Michel Alexeev** 
* **Benjamin Souane** 


## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
