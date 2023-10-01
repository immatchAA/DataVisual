#!/usr/bin/env python
# coding: utf-8

# Data Visualization Project

# In[4]:


import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt 
pd.plotting.register_matplotlib_converters()
get_ipython().run_line_magic('matplotlib', 'inline')
#import seaborn as sns


# In[5]:


df = pd.read_csv('WHO-COVID-19-global-data.csv')
df.head(5)


# In[6]:


countries_of_interest = ['Japan', 'New Zealand', 'Philippines']

filtered_df = df[df['Country'].isin(countries_of_interest)]

date_reported_values = filtered_df['Date_reported'].unique()
for date in date_reported_values:
    print(date)


# In[8]:


df = pd.read_csv('WHO-COVID-19-global-data.csv')
filtered_data = df[df['WHO_region'] == 'WPRO']
filtered_data = filtered_data.reset_index(drop=True)


# In[9]:


# Country lists
Country1 = ['Japan']
Country2 = ['Philippines']
Country3 = ['New Zealand']


japan_countries_wpro_data = df[(df['Country'].isin(Country1)) & (df['WHO_region'] == 'WPRO')]
philippines_countries_wpro_data = df[(df['Country'].isin(Country2)) & (df['WHO_region'] == 'WPRO')]
nz_countries_wpro_data = df[(df['Country'].isin(Country3)) & (df['WHO_region'] == 'WPRO')]


japan_deaths = japan_countries_wpro_data['Cumulative_deaths'].sum()
japan_cases = japan_countries_wpro_data['Cumulative_cases'].sum()
jp_difference = japan_cases - japan_deaths


PH_deaths = philippines_countries_wpro_data['Cumulative_deaths'].sum()
PH_cases = philippines_countries_wpro_data['Cumulative_cases'].sum()
PH_difference = PH_cases - PH_deaths


NZ_deaths = nz_countries_wpro_data['Cumulative_deaths'].sum()
NZ_cases = nz_countries_wpro_data['Cumulative_cases'].sum()
NZ_difference= NZ_cases - NZ_deaths

total_deaths = japan_deaths + PH_deaths + NZ_deaths
total_cases = japan_cases + PH_cases + NZ_cases


percentage_japandeaths = (japan_deaths / total_deaths) * 100
percentage_japancases = (japan_cases / total_cases) * 100

percentage_PHdeaths = (PH_deaths / total_deaths) * 100
percentage_PHcases = (PH_cases / total_cases) * 100

percentage_NZdeaths = (NZ_deaths / total_deaths) * 100
percentage_NZcases = (NZ_cases / total_cases) * 100

#total for all three countries
print(f"Total Deaths for three Countries: {total_deaths}")
print(f"Total Cases for three Countries: {total_cases}")
print(f"Total Percentage of Deaths for Three Countries: {total_deaths:.2f}%")
print(f"Total Percentage of Cases for Three Countries: {total_cases:.2f}%")


#individual country data
print("\nIndividual Country Data:")
print(f"\n{', '.join(Country1)} (WPRO):")
print(f"Total Cumulative Deaths: {japan_deaths}")
print(f"Total Cumulative Cases: {japan_cases}")
print(f"Death Percentage: {percentage_japandeaths:.2f}%")
print(f"Case Percentage: {percentage_japancases:.2f}%")
print(f"Difference between Total Cases and Total Deaths: {jp_difference}")

print(f"\n{', '.join(Country2)} (WPRO):")
print(f"Total Cumulative Deaths: {PH_deaths}")
print(f"Total Cumulative Cases: {PH_cases}")
print(f"Death Percentage: {percentage_PHdeaths:.2f}%")
print(f"Case Percentage: {percentage_PHcases:.2f}%")
print(f"Difference between Total Cases and Total Deaths: {PH_difference}")

print(f"\n{', '.join(Country3)} (WPRO):")
print(f"Total Cumulative Deaths: {NZ_deaths}")
print(f"Total Cumulative Cases: {NZ_cases}")
print(f"Death Percentage: {percentage_NZdeaths:.2f}%")
print(f"Case Percentage: {percentage_NZcases:.2f}%")
print(f"Difference between Total Cases and Total Deaths: {NZ_difference}")


# In[14]:


import streamlit as st

st.title('My Dashboard')
st.write('This is a simple Streamlit app!')


# In[10]:


#pie chart
mylabels = ['Japan', 'Philippines', 'New Zealand']

x = [japan_deaths, PH_deaths, NZ_deaths]
y = [japan_cases, PH_cases, NZ_cases]
differences = [jp_difference, PH_difference, NZ_difference]

myexplode = [0.2, 0.2, 0.2]
mycolors1 = ["Bisque", "Orange", "Yellow"]
mycolors2= ["OrangeRed", "Tomato", "PaleVioletRed"]
mycolors3 = ["Pink", "Plum", "RebeccaPurple"]

fig, axes = plt.subplots(nrows = 1, ncols=3, figsize=(10, 8))
fig.patch.set_facecolor('SeaShell')

axes[0].pie(x, labels= mylabels, autopct='%1.1f%%', startangle=90, colors = mycolors1)
axes[0].set_title('Total Deaths' , color = 'black')

axes[1].pie(y, labels= mylabels, autopct='%1.1f%%', startangle=90, colors = mycolors2)
axes[1].set_title('Total Cases', color = 'black')

axes[2].pie(differences, labels=mylabels, explode = myexplode, autopct='%1.1f%%', startangle=90, colors=mycolors3)
axes[2].set_title('Case - Death Difference', color='black')

#plt.pie(differences, explode = myexplode, autopct='%1.1f%%', shadow=True, colors = mycolors3)
plt.tight_layout
plt.legend(loc = 'right')
plt.show()
st.pyplot(fig)


# In[13]:


df['Date_reported'] = pd.to_datetime(df['Date_reported'])

#PH2020
start_date = '2020-01-01'
end_date = '2020-12-31'
country = 'Philippines'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

PH_deaths_2020 = filtered_data['Cumulative_deaths'].sum()
PH_cases_2020 = filtered_data['Cumulative_cases'].sum()

print(f"Date Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {PH_deaths_2020}")
print(f"Total Cases: {PH_cases_2020}")

#PH2021

start_date = '2021-01-01'
end_date = '2021-12-31'
country = 'Philippines'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

PH_deaths_2021 = filtered_data['Cumulative_deaths'].sum()
PH_cases_2021 = filtered_data['Cumulative_cases'].sum()

print(f"\nDate Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {PH_deaths_2021}")
print(f"Total Cases: {PH_cases_2021}")

#PH2022

start_date = '2022-01-01'
end_date = '2022-12-31'
country = 'Philippines'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

PH_deaths_2022 = filtered_data['Cumulative_deaths'].sum()
PH_cases_2022 = filtered_data['Cumulative_cases'].sum()

print(f"\nDate Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {PH_deaths_2022}")
print(f"Total Cases: {PH_cases_2022}")

#PH2023

start_date = '2023-01-01'
end_date = '2023-12-31'
country = 'Philippines'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

PH_deaths_2023 = filtered_data['Cumulative_deaths'].sum()
PH_cases_2023 = filtered_data['Cumulative_cases'].sum()


print(f"\nDate Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {PH_deaths_2023}")
print(f"Total Cases: {PH_cases_2023}")





# In[15]:


df['Date_reported'] = pd.to_datetime(df['Date_reported'])

#JP2020
start_date = '2020-01-01'
end_date = '2020-12-31'
country = 'Japan'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

JP_deaths_2020 = filtered_data['Cumulative_deaths'].sum()
JP_cases_2020 = filtered_data['Cumulative_cases'].sum()

print(f"Date Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {JP_deaths_2020}")
print(f"Total Cases: {JP_cases_2020}")

#JP2021

start_date = '2021-01-01'
end_date = '2021-12-31'
country = 'Japan'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

JP_deaths_2021 = filtered_data['Cumulative_deaths'].sum()
JP_cases_2021 = filtered_data['Cumulative_cases'].sum()

print(f"\nDate Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {JP_deaths_2021}")
print(f"Total Cases: {JP_cases_2021}")

#JP2022

start_date = '2022-01-01'
end_date = '2022-12-31'
country = 'Japan'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

JP_deaths_2022 = filtered_data['Cumulative_deaths'].sum()
JP_cases_2022 = filtered_data['Cumulative_cases'].sum()

print(f"\nDate Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {JP_deaths_2022}")
print(f"Total Cases: {JP_cases_2022}")

#JP2023

start_date = '2023-01-01'
end_date = '2023-12-31'
country = 'Japan'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

JP_deaths_2023 = filtered_data['Cumulative_deaths'].sum()
JP_cases_2023 = filtered_data['Cumulative_cases'].sum()

print(f"\nDate Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {JP_deaths_2023}")
print(f"Total Cases: {JP_cases_2023}")


# In[37]:


df['Date_reported'] = pd.to_datetime(df['Date_reported'])

#NZ2020
start_date = '2020-01-01'
end_date = '2020-12-31'
country = 'New Zealand'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

NZ_deaths_2020 = filtered_data['Cumulative_deaths'].sum()
NZ_cases_2020 = filtered_data['Cumulative_cases'].sum()

print(f"Date Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {NZ_deaths_2020}")
print(f"Total Cases: {total_cases}")

#NZ2021

start_date = '2021-01-01'
end_date = '2021-12-31'
country = 'New Zealand'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

NZ_deaths_2021 = filtered_data['Cumulative_deaths'].sum()
NZ_cases_2021 = filtered_data['Cumulative_cases'].sum()

print(f"\nDate Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {NZ_deaths_2021}")
print(f"Total Cases: {NZ_cases_2021}")

#NZ2022

start_date = '2022-01-01'
end_date = '2022-12-31'
country = 'New Zealand'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

NZ_deaths_2022 = filtered_data['Cumulative_deaths'].sum()
NZ_cases_2022 = filtered_data['Cumulative_cases'].sum()

print(f"\nDate Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {NZ_deaths_2022}")
print(f"Total Cases: {NZ_cases_2022}")

#NZ2023

start_date = '2023-01-01'
end_date = '2023-12-31'
country = 'New Zealand'

filtered_data = df[(df['Date_reported'] >= start_date) & (df['Date_reported'] <= end_date) & (df['Country'] == country)]

NZ_deaths_2023 = filtered_data['Cumulative_deaths'].sum()
NZ_cases_2023 = filtered_data['Cumulative_cases'].sum()

print(f"\nDate Range: {start_date} to {end_date}")
print(f"Country: {country}")
print(f"Total Deaths: {NZ_deaths_2023}")
print(f"Total Cases: {NZ_cases_2023}")


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

data = {
    'Year': [2020, 2021, 2022, 2023],
    'Japan Deaths': [357140, 4800715, 12617383, 19203785],
    'Philippines Deaths': [1039042, 9804094, 22025430, 17509931],
    'New Zealand Deaths': [5971, 10492, 475841, 753550]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the background color to black
plt.style.use('dark_background')

# Create a line graph with different colors for each country
plt.figure(figsize=(10, 6))

plt.plot(df['Year'], df['Japan Deaths'], marker='o', label='Japan', color='red')
plt.plot(df['Year'], df['Philippines Deaths'], marker='o', label='Philippines', color='green')
plt.plot(df['Year'], df['New Zealand Deaths'], marker='o', label='New Zealand', color='blue')

plt.xlabel('Year')
plt.ylabel('Total Deaths')
plt.title('Total COVID-19 Deaths in three Countries (2020-2023)')
plt.legend()
plt.grid(True)

plt.show()

st.pyplot(fig)


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

bar_width = 0.25

Philippines = [53051873, 579175132, 1371528925, 1080832185]
Japan = [18192639, 366241338, 4740416950, 8796028378]
NewZealand = [18037757260, 1467087, 433795324,596651546]

br1 = np.arange(len(Philippines))
br2 = [x + bar_width for x in br1]
br3 = [x + bar_width for x in br2]

fig, ax = plt.subplots(figsize=(12, 8))  

plt.bar(br1, Philippines, color='r', width=bar_width, edgecolor='grey', label='Philippines')
plt.bar(br2, Japan, color='g', width=bar_width, edgecolor='grey', label='Japan')
plt.bar(br3, NewZealand, color='b', width=bar_width, edgecolor='grey', label='New Zealand')

plt.xlabel('Year', fontweight='bold', fontsize=15)
plt.ylabel('Total Cases', fontweight='bold', fontsize=15)
plt.xticks([r + bar_width for r in range(len(Philippines))], ['2020', '2021', '2022', '2023'])

plt.legend()
plt.show()
st.pyplot(fig)


# In[ ]:




