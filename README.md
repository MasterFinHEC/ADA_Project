# ADA_Project
Repository for the Advanced Data Analytics project from the Master of science in finance from HEC Lausanne. 

![HEC Lausanne Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/HEC_Lausanne_logo.svg/293px-HEC_Lausanne_logo.svg.png)

## Project
The project is to try and find good predictors of a pandemic reach on a country. 

Therefore using data on : 

- Economic Activity (GDP per capita)
- Level of democracy
- Density of populaiton
- Inequality (Gini Index)
- Quality of health System
- Openness Index (typically Trade/GDP data)

We want to see if there is a way to predict the way a country will be affected by a pandemic using the Covid-19 data. 

The analysis is perform in two step : 

1. Model the pandemic dynamic with a logistic model:

![Logistic Equation](https://wikimedia.org/api/rest_v1/media/math/render/svg/9e26947596d387d045be3baeb72c11270a065665)

2. Find the relation between the logistic parameter and the countries predictors.

### Data

#### Covid - 19 Data
Taken directly from the John Hopkins for decease center, they are updated every day when running the code.

You can get them *[here](https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/).

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

1. Democracy Index from [The Economist Intelligence Unit](https://www.eiu.com/topic/democracy-index).
2. A Data set mapping the countries to the continents. 
3. Politics and government index from World Data.

### Implementation

In order to run the code, you need to install the following libraries on your computer: 

```
pip install plotly
pip install tensorflow
```
Then the implementation runs relatively fast. However, the "the logistic" takes aroung an hour. Therefore, the optimal weights are already loaded in the "Optimal" folder and the code will download them. If you want to run the optimisation yourself, you just need to change the property of the cells from "Raw" to "Code".

## Authors

* **Alessia Di Pietro** 
* **Antoine-Michel Alexeev** 
* **Benjamin Souane** 


## Acknowledgments

* Prof. [Simon Scheidegger](https://sites.google.com/site/simonscheidegger/home?authuser=0)
* Antoine Didisheim
* Hat tip to anyone whose code was used
