import streamlit as st
import pickle
import pandas as pd

teams=['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals']

pipe = pickle.load(open('pipe.pkl','rb'))


st.title('IPL WIN PREDICTOR')

# Load data to get city-venue mapping
match = pd.read_csv('matches.csv')
cities = sorted(match['city'].dropna().unique())
city_venue = match.groupby('city')['venue'].unique().apply(list).to_dict()

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

city = st.selectbox('Select the city', cities)

# Dependent dropdown for venue
if city in city_venue:
    selected_venue = st.selectbox('Select the venue', sorted(city_venue[city]))
else:
    st.selectbox('Select the venue', [])

target=st.number_input('Target')

col3, col4, col5= st.columns(3)

with col3:
    score=st.number_input('Score')

with col4:
    overs=st.number_input('Overs completed')

with col5:
    wickets=st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left=target-score
    balls_left=120-(overs*6)
    wickets=10-wickets
    crr=score/overs
    rrr = 0
    if balls_left > 0:
        rrr = (runs_left * 6) / balls_left

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result=pipe.predict_proba(input_df)
    loss=result[0][0]
    win=result[0][1]

    st.header(batting_team + " - " + str(round(win*100)) + "%")
    st.header(bowling_team + " - " + str(round(loss*100)) + "%")
