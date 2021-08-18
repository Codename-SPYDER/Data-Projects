import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.title('NFL Football Stats (Rushing) Explorer')

st.markdown("""
	This app performs simple webscraping of NFL Football player stats data (focusing on Rushing)
	""")


st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990, 2020))))

#Web scraping NFL player stats
#https://www.pro-football-reference.com/years/2019/rushing.htm 
@st.cache
def load_data(year):
	url = "https://www.pro-football-reference.com/years/" + str(year) + "/rushing.htm"
	html = pd.read_html(url, header = 1)
	df = html[0]
	raw = df.drop(df[df.Age == 'Age'].index) # Deletes repeating hear=ders in content
	raw = raw.fillna(0)
	playerstats = raw.drop(['Rk'], axis=1)
	return playerstats
playerstats = load_data(selected_year)

#Sidebar - Team selection
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# Sidebar - Position selection
unique_pos = ['RB', 'QB', 'WR', 'FB', 'TE']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

st.header('Display Player Stats of Selected Team(s)')
st.write('Data Dimension: ' + str(df_selected_team.shape[0]) + ' rows and ' + str(df_selected_team.shape[1]) + ' columns.')
st.dataframe(df_selected_team)

# Download NFL Data

def filedownload(df):
	csv = df.to_csv(index=False)
	b64 = base64.b64encode(csv.encode()).decode() # strings to bytes conversion
	href = f'<a href="data:file/csv;base64,{b64}" download = "playerstats.csv">Download CSV File</a>'
	return href

st.markdown(filedownload(df_selected_team), unsafe_allow_html=True)