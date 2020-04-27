#!/usr/bin/env python
# coding: utf-8

# # Importing NumPy and Pandas Library

# #### NumPy is a Linear Algebra Library in Python which is used to perform various numerical operations, Linear Algebra operations.
# #### NumPy Library provides us the facility to perform basic operations on the datasets.
# #### Pandas Library is a very important library and is used for fast analysis, data cleaning, and data preparation.
# #### Pandas allow us to load data from various sources and to deal with them in tabular fashion.

# In[1]:


import numpy as np
import pandas as pd


# #### Note: NumPy is the most important library for the PyData Ecosystem because for almost all other libraries it acts as a building block.

# # Importing Matplotlib and Seaborn Library

# #### Matplotlib is a Data Visualisation Library in Python. It is also known as Grandfather Library for all other Visualisation libraries in Python. It provides us great controls on almost every aspect of the figure.
# #### Seaborn is an Advance Data Visualisation Library in Python and built on top of the Matplotlib. It is a Statistical Ploting Library and it has beautiful default styles.
# #### Seaborn Library works very well with Pandas data frame objects.

# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Note: This inline statement is only used when working with jupyter notebook, and is used to see the figure inside the notebook.

# # Importing Plotly and Cufflinks Library

# #### Plotly is an Interactive Visualisation Library in Python. It is an open-source library. Plotly is a company a website that allows us to create interactive figures for our visualization purpose both online and offline.
# #### Cufflinks connect plotly with pandas.

# In[3]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf


# #### init_notebook_mode allows us to connect javascript with our notebook so that we can use those plotting features in our notebook.

# In[4]:


init_notebook_mode(connected=True)


# #### cf.go_offline( ) allows us to use cufflinks offline

# In[5]:


cf.go_offline()


# # 1. Analysing the Present Condition in India

# ## 1.1 Reading the DataSets

# #### Note: index_col is used to make the first column of our dataset as an index for our DataFrame object

# In[6]:


ind_state = pd.read_excel('India_covid19_states_wise.xlsx', index_col=0)


# In[7]:


ind_daily = pd.read_excel('per_day_cases.xlsx')


# In[8]:


ind_complete = pd.read_csv('https://raw.githubusercontent.com/covid19india/api/gh-pages/csv/latest/raw_data.csv', index_col = 0)


# In[9]:


ind_state.info()


# In[10]:


ind_daily.info()


# In[11]:


ind_complete.info()


# In[12]:


ind_state.head()


# In[13]:


ind_daily.head()


# In[14]:


ind_complete.head()


# In[15]:


ind_complete.drop(['Estimated Onset Date', 'Notes', 'Contracted from which Patient (Suspected)', 'Source_1', 'Source_2', 'Source_3'], axis = 1, inplace = True)


# #### Note: drop method is used to drop the columns from our DataFrame
# 
# We are performing this step because we don't need these columns for our analysis

# In[16]:


ind_complete.head()


# ## 1.2 Working with ind_complete DataFrame for cleaning the data

# #### Analyzing the null values in the dataset

# In[17]:


plt.figure(figsize = (12,8))
sns.heatmap(ind_complete.isnull(), yticklabels=False, cbar = False, cmap='viridis')


# #### yellow lines represent the null values
# #### By analyzing the heatmap we can see that there are 8 columns out of 13 having so many null values Age and Gender are the two columns which we can't remove since we need them for analyzing the age and gender of the patients other than these columns we will remove all other columns.

# In[18]:


ind_complete.drop(['State Patient Number', 'Detected City', 'Detected District', 'Nationality', 
                   'Type of transmission', 'Backup Notes'], axis = 1, inplace = True)


# #### Our final dataset ind_complete for furthur use

# In[19]:


ind_complete.head()


# ## 1.3 Analysing COVID19 Cases in India
# ## Total Number of Confirmed cases in India

# In[20]:


total = ind_daily.iloc[-1,1]
print("Total number of Confirmed cases till 24th April 2020 in India: {}".format(total))


# #### This color chart represents the level of Danger in respective states
# Darker the Red Color, more dangerous is the situation

# In[21]:


ind_state.drop('Cured / Discharged', axis = 1).style.background_gradient(cmap = 'Reds')


# ## Counting the Active cases, cured and deaths

# In[22]:


active = ind_daily.iloc[-1,3]
cured = sum(ind_daily['Daily Recovery'])
deaths = ind_daily.iloc[-1,4]
print("Total number of Currently Active cases in India: {}".format(active))
print("Total number of Cured/Discharged cases in India: {}".format(cured))
print("Total number of Deaths in India: {}".format(deaths))


# In[23]:


Tot_cases = ind_state.groupby('Name of State / UT')['Active Cases'].sum().sort_values(ascending = False).to_frame()
Tot_cases.style.background_gradient(cmap = 'Reds')


# ## Number of Total cases, Active and Cured cases
# using Barplot with seaborn

# In[24]:


plt.figure(figsize = (18,35))
data = ind_state.copy()
data.sort_values('Total Confirmed Cases', ascending = False, inplace = True)

#sns.set_color_codes('pastel')
sns.barplot(y = 'Name of State / UT', x = 'Total Confirmed Cases', data = data, label = 'Total', color = 'r', saturation= 20)

sns.barplot(x = 'Active Cases', y = 'Name of State / UT', data = data, label = 'Active', color = 'b')

sns.barplot(x = 'Cured / Discharged', y = 'Name of State / UT', data = data, label = 'Cured', color = 'g')

plt.legend(loc = 'lower right', ncol = 3, frameon = True)


# ## Total COVID19 Cases per state
# using Barplot

# In[25]:


df = px.data.tips()
fig = px.bar(ind_state.sort_values(by = 'Total Confirmed Cases'), x = 'Name of State / UT', 
             y = 'Total Confirmed Cases', width = 1000, height = 700, 
             title = 'Total COVID19 Cases per State', color='Total Confirmed Cases', 
             hover_data=['Total Confirmed Cases', 'Active Cases', 'Cured / Discharged', 'Deaths'],
             color_continuous_scale='Bluered_r')
fig.show()


# ## Total COVID19 Cases in India
# using Scatter Plot

# In[26]:


fig = px.scatter(ind_daily, x = 'Date', y = 'Total Cases', color = 'Total Cases', color_continuous_scale='Bluered_r',
                title = 'Total COVID19 Cases in India', width = 1000, height = 700)
fig.show()


# ## Daily New Cases in India
# using Bar Plot

# In[27]:


fig = px.bar(ind_daily, x = 'Date', y = 'Daily New Cases', title = 'Daily New Cases in India', color = 'Daily New Cases',
            color_continuous_scale='Bluered_r', width = 1000, height = 700)
fig.show()


# ## Active Cases in India
# using Scatter Plot

# In[28]:


fig = px.scatter(ind_daily, x = 'Date', y = 'Active Cases', color = 'Active Cases', color_continuous_scale='Bluered_r', 
                title = 'Active Cases in India', width = 1000, height = 700)
fig.show()


# ## Total Coronavirus Deaths in India
# using Scatter Plot

# In[29]:


fig = px.scatter(ind_daily, x = 'Date', y = 'Total Deaths', color = 'Total Deaths', color_continuous_scale='Bluered_r',
                title = 'Total Deaths', width = 1000, height = 700)
fig.show()


# ## Daily New Deaths in India
# using Bar Plot 

# In[30]:


fig = px.bar(ind_daily, x = 'Date', y = 'Daily Deaths', color = 'Daily Deaths', color_continuous_scale='Bluered_r',
            title = 'Daily Deaths', width = 1000, height = 700)
fig.show()


# ## Newly Infected vs Newly Recovered vs Newly Death in India

# In[31]:


ind_daily_c = ind_daily.copy()
ind_daily_c.set_index('Date', inplace = True)
ind_daily_c.drop(['Total Cases', 'Active Cases', 'Total Deaths'], axis = 1).iplot(mode = 'markers+lines', size = 5,
                                            yTitle = 'Daily new Coronavirus Cases + Cured + Deaths', title = 'New Cases vs New Recoveries vs New Deaths')


# ## Creating temperary DataFrame for age and gender
# This is important because if we try to perform following required operations directly on a original dataframe then it will cause us to loose some important informations. Because there are some rows which either has Age value to be Null or Gender value to be Null and using dropna directly will lead us to loose some important informations.

# In[32]:


ind_complete.columns


# In[33]:


temp_age = pd.DataFrame(ind_complete['Age Bracket'])
temp_gender = pd.DataFrame(ind_complete['Gender'])


# #### This step is performed for dropping all the null value rows

# In[34]:


temp_age.dropna(inplace = True)
temp_gender.dropna(inplace = True)


# #### This will add the Count column in our temperary dataframe
# We need this column to count the instances

# In[35]:


temp_age['Count'] = 1
temp_gender['Count'] = 1


# ## Patients by Gender

# In[36]:


total = sum(temp_gender['Count'])
fig = px.histogram(temp_gender, x = 'Gender', y = 'Count', 
                   title = 'Sample size: {} patients'.format(total), color = 'Gender', width = 1000, height = 700)
fig.show()


# ## Patients by Age
# The following steps are required to replace the given range with the value of our choice within the given range

# In[37]:


temp_age[temp_age['Age Bracket'] == '28-35']


# In[38]:


temp_age['Age Bracket'][925] = 31
temp_age['Age Bracket'][926] = 31
temp_age['Age Bracket'][927] = 31
temp_age['Age Bracket'][928] = 31


# In[39]:


temp_age[temp_age['Age Bracket'] == '28-35']


# #### Method dtype shows that the Age is a string and to do furthur process we need to convert it into float. And to do this we'll use astype method

# In[42]:


temp_age.dtypes


# In[43]:


temp_age = temp_age.astype({'Age Bracket': 'float64'})


# In[44]:


temp_age.dtypes


# #### This function is created to count the number of patients between a certain range of age and then store it in a list which will be used to create a new dataframe for ploting our histogram
# A new dataframe df is used to plot histogram 

# In[45]:


age_sum = [0,0,0,0,0,0,0,0,0,0]
j = 0
for i in range(0, 100, 10):
    c = temp_age[(temp_age['Age Bracket'] < (i+11)) & (temp_age['Age Bracket'] > i)].count()
    age_sum[j] = c['Count']
    j += 1


# In[46]:


age_group = ['0 - 10', '11-20', '21 - 30', '31 - 40', '41 - 50', '51 - 60', '61 - 70', '71 - 80', '81 - 90', '91 - 100']


# In[47]:


df = pd.DataFrame(data = age_sum, columns = ['Count'])


# In[48]:


df['Age Group'] = age_group


# In[49]:


df


# In[50]:


total = sum(df['Count'])
fig = px.bar(df, y = 'Count', x = 'Age Group', color = 'Age Group', title = 'Sample Size: {} patients'.format(total), 
             width = 1000, height = 700)
fig.show()


# # 2. Predicting and Forecasting number of cases in India

# ## Prophet
# Prophet is an open source software released by Facebook's core Data Science team. It is available for download on CRAN and PYPI.
# 
# We use Prophet, a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.
# ## Why Prophet?

# #### Accurate and Fast: 
# Prophet is used in many applications across Facebook for producing reliable forecasts for planning and goal setting. Facebook finds it to perform better than any other approach in the majority of cases. It fit models in Stan so that you get forecasts in just a few seconds.

# #### Fully Automatic:
# Get a reasonable forecast on messy data with no manual effort. Prophet is robust to outliers, missing data and dramatic changes in your time series.

# #### Tunable Forecast:
# The Prophet procedure includes many possibilities for users to tweak and adjust forecasts. You can use human interpretable parameters to improve your forecast by adding your domain knowledge.

# #### Available in R and Python
# Facebook has implemented the Prophet procedures in R and Python. Both of them share the same underlying Stan code for fitting. You can use whatever language you are comfortable with to get forecasts.
# 
# #### We are using Prophet because in majority of the cases it works best with time series datas.

# In[51]:


from fbprophet import Prophet


# ## 2.1 Growth in Number of Patients per day

# In[52]:


fig = px.bar(ind_daily, x = 'Date', y = 'Total Cases', title = 'Daily Growth in number', width = 1000, height = 700)
fig.show()


# ## 2.2 Prediction

# In[53]:


Total_cases = ind_daily[['Date', 'Total Cases']]


# The input to a Prophet is always a dataframe with two columns: ds and y. The ds(Datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric and represents the measurements we wish to forecast.

# In[54]:


Total_cases.head()


# In[55]:


Total_cases.columns = ['ds', 'y']


# In[56]:


Total_cases['ds'] = pd.to_datetime(Total_cases['ds'])


# In[57]:


Total_cases.head()


# #### Generating a week ahead forecast of confirmed cases in India regarding COVID19 using Prophet with 95% prediction interval

# In[ ]:


m = Prophet(interval_width=0.95)


# interval_width is actually confidence. This is given in a decimal.
# 
# 0.95 actually means that there is still scope for 5% error and it means that we are 95% sure that our forecast is correct.
# 

# In[ ]:


m.fit(Total_cases)


# In[ ]:


future = m.make_future_dataframe(periods=7)
future.tail(7)


# Here future is the new dataframe which will store the result of the predictions
# 
# In the above step, we are adding new rows to store the results in the respective rows

# The predict method will assign each row in future a predicted value which is named as yhat. If you pass in historical dates, it will provide an in-sample fit. The forecast object here is a new dataframe that includes a column yhat with the forecast, as well as columns for components and uncertainty intervals.

# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(8)


# where yhat_lower tells the lowest possible value and yhat_upper tells the highest possible value, basically they tell the range
# 
# and according to our model they tell the range within which the count of the total infections will be at the respective date

# In[ ]:


recovered_forecast_plot = m.plot(forecast)


# In[ ]:


recovered_forecast_plot = m.plot_components(forecast)

