import streamlit as st
import numpy as np
import pandas as pd
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
import altair as alt
import datetime
from model_container import ModelContainer
import date_formater as dtf

import bar_chart as bc

class App():
    def __init__(self):
        self.model = ModelContainer()
        self.model.load_model(model='exports/models/model1.pkl',features='exports/models/features1.csv')
        self.genre_inputs = dict([(x.replace('Genre_',''),x) for x in self.model.features if 'Genre_' in x])
        self.venue_inputs = dict([(x.replace('Venue_',''),x) for x in self.model.features if 'Venue_' in x])
        self.postype_inputs = dict([(x.replace('PostType_',''),x) for x in self.model.features if 'PostType_' in x])
        #self.platform_inputs = [(x, x-'Genre_') for x in self.model.features if x == 'INSTA' or x == 'FB']
        self.artist_inputs = dict([(x.replace('Artist_',''),x) for x in self.model.features if 'Artist_' in x])
        self.merged_df = None
        self.inputs = {
            'artist': None,
            'genre': None,
            'venue': None,
            'post-type': None,
            'event-date': None
        }

    def update_progress(self,i):
        self.progress_bar.progress(i)

    def model_prediction(self,model,inputs):
        print(self.inputs)
        if any(v is None for v in self.inputs.values()):
            self.merged_df = None
            return False
        self.progress_bar = st.progress(0)

        X_simulation = {
            'FB': self.model.generate_simulation(inputs['artist'],inputs['genre'],inputs['venue'],inputs['post-type'],'FB',inputs['event-date'],self.update_progress),
            'INSTA': self.model.generate_simulation(inputs['artist'],inputs['genre'],inputs['venue'],inputs['post-type'],'INSTA',inputs['event-date'],self.update_progress)
        }

        dates = {
            'FB': X_simulation['FB'][0],
            'INSTA': X_simulation['INSTA'][0]
        }

        y_simulation = {
            'FB': pd.concat([ dates['FB'].reset_index(drop=True), self.model.predict(X_simulation['FB'][1]).reset_index(drop=True) ], axis=1),
            'INSTA': pd.concat([ dates['INSTA'].reset_index(drop=True), self.model.predict(X_simulation['INSTA'][1]).reset_index(drop=True) ], axis=1)
        }
    
        df1 = pd.DataFrame(y_simulation['FB'])
        df2 = pd.DataFrame(y_simulation['INSTA'])

        self.merged_df = df1.merge(df2, on='Dates', suffixes=('FB', 'INSTA'))
        self.update_progress(100)
        if self.merged_df is not None:
            self.bar_charts()
        #self.draw()

    def bar_charts(self):
        #st.header('Theater de Veste')
        #st.header('Social Media Timing Optimization')

        chart_fb = bc.bar_plot(self.merged_df[ self.merged_df['PredFB'].isin([max(self.merged_df['PredFB']),min(self.merged_df['PredFB'])]) ],platform='Facebook')
        chart_insta = bc.bar_plot(self.merged_df[ self.merged_df['PredINSTA'].isin([max(self.merged_df['PredINSTA']),min(self.merged_df['PredINSTA'])]) ],platform='Instagram')

        #min_chart_fb = bc.bar_plot(self.merged_df[self.merged_df['PredFB'] == min(self.merged_df['PredFB'])],platform='Facebook')
        #min_chart_insta = bc.bar_plot(self.merged_df[self.merged_df['PredINSTA'] == max(self.merged_df['PredINSTA'])],platform='Instagram')

        col1, col2 = st.columns(2)
        with col1:
            chart_fb
        with col2:
            chart_insta
        self.progress_bar.empty()

    def set_inputs(self,a,b,c,d,e):
        print('hello')
        self.inputs['artist'] = a
        self.inputs['genre'] = b
        self.inputs['venue'] = c
        self.inputs['post-type'] = d
        self.inputs['event-date'] = e
        #self.model_prediction(self.model,self.inputs)

    def draw(self):
        with st.sidebar:
            st.header('Theater De Vest')
            st.header('Social Media Marketing Predictions')
            st.title('')

            self.inputs['artist'] = self.artist_inputs[st.selectbox('Gezelschap: ',self.artist_inputs.keys())] 
            self.inputs['genre'] = self.genre_inputs[st.selectbox('Genre: ',self.genre_inputs.keys())]
            self.inputs['venue'] = self.venue_inputs[st.selectbox('Zaal: ',self.venue_inputs.keys())]
            self.inputs['post-type'] = self.postype_inputs[st.selectbox('Post Type: ',self.postype_inputs.keys())]
            self.inputs['event-date'] = self._datetime_generator(st.date_input("Event Date: ", datetime.date(2019, 7, 6)),st.time_input('Event Time:', datetime.time(8, 45)))
            self.button = st.button('Predict',on_click=lambda: self.model_prediction(self.model,self.inputs))

    def _datetime_generator(self,date,time):
        event_date = str(date) + ' ' + str(time)
        return event_date



app = App()
app.draw()