import pandas as pd
import numpy as np
import date_formater as dtf
import altair as alt

def bar_plot(df,ids='Dates',values=['FB','INSTA'],platform='Facebook'):

    df = df.rename(columns={'PredFB':'FB','PredINSTA':'INSTA'})
    prediction_table = pd.melt(df, id_vars=ids, value_vars=values)

    domain = ["FB", "INSTA"]
    range_ = ['#1877F2','#E95950']

    chart = alt.Chart(
        prediction_table,
        title=f'Best Date on {platform}'
    ).mark_bar(
        cornerRadius=10,
        opacity=1,
    ).encode(
        column = alt.Column('Dates', spacing = 30, header = alt.Header(labelOrient = "bottom")),
        x = alt.X('variable:N', sort = ["FB", "INSTA"],  axis=None),
        y = alt.Y('value:Q', title='Tickets / Capacity', scale=alt.Scale(domain=[0.0, 1.0])),
        color = alt.Color('variable').scale(domain=domain, range=range_)
    ).configure_view(
        stroke='transparent'
    )
    return chart