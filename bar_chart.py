import pandas as pd
import numpy as np
import date_formater as dtf
import altair as alt

def bar_plot(df,capacity,ids='Dates',values=['FB','INSTA'],platform='Facebook'):

    df = df.rename(columns={'PredFB':'FB','PredINSTA':'INSTA'})
    prediction_table = pd.melt(df, id_vars=ids, value_vars=values)

    #prediction_table['Time'] = prediction_table['Dates'].apply(dtf.to_time)

    capacity = capacity if capacity > 1 else 1
    prediction_table['value'] = prediction_table['value'] * capacity
    if capacity > 1:
        prediction_table['value'] = prediction_table['value'].round(0)

    domain = ["FB", "INSTA"]
    range_ = ['#1877F2','#E95950']

    chart = alt.Chart(
        prediction_table,
        title=f'Beste Datum & Tijd voor {platform}'
    ).mark_bar(
        cornerRadius=10,
        opacity=1,
    ).encode(
        column = alt.Column('Dates:T', spacing = 50, timeUnit='utcyearmonthdatehoursminutes', header = alt.Header(title='Verkoop Vergelijking', titleOrient='bottom', labelOrient='bottom',labelAlign='center', labelPadding=30, labelAngle=-12)),
        x = alt.X('variable:N', title='', sort = ["FB", "INSTA"], axis=None),
        y = alt.Y('value:Q', title='Verwachte Kaartverkoop', scale=alt.Scale(domain=[0.0, 1.0*capacity])) if capacity > 1 \
            else alt.Y('value:Q', title='Verwachte ( Kaartverkoop / Capaciteit )', scale=alt.Scale(domain=[0.0, 1.0*capacity])),
        color = alt.Color('variable:N').scale(domain=domain, range=range_)
    ).configure_view(
        stroke='transparent'
    )
    return chart