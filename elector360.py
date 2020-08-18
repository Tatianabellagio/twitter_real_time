import dash
from datetime import datetime as dt
import dash_html_components as html
import dash_core_components as dcc
import pandas as pd
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import numpy as np
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import networkx as nx
import dash_bootstrap_components as dbc
from ast import literal_eval
import re
from unidecode import unidecode
import matplotlib.pyplot as plt
import geopandas as gpd
from urllib.request import urlopen

from urllib.request import urlopen
import json
external_stylesheets = [dbc.themes.BOOTSTRAP,]



#preprocesamiento 
def get_hashtags(x, hashtags):
    
    list_hashtags = [token for token in x if str(token) in hashtags]
    if len(list_hashtags) > 0:
        return True
    else:
        return False
    
mapbox_accesstoken = 'pk.eyJ1IjoibWF0aWFzY29yZWJpIiwiYSI6ImNrYWNwMHJyajBnOGEycnRieTZ6ZGQ5cmEifQ.JUkB1ax6L574PG33zwQ5iQ'

stopwords = set(['lo', 'me', 'que', 'se', 'nos', 'le', 'y', 'a', ' ', '', 'o'])

df = pd.read_csv('df_total_27_05_2020.csv',  low_memory=False)

df = df.drop_duplicates(subset=['id', 'conversation_id', 'tweet', 'user_id', 'user_rt_id',  'retweet_id', 'neighborhood'])

df = df[df['trans_src']=='es']
df['final_date'] = np.where(df['retweet']==True, df['retweet_date'], df['date'])

df['final_date'] = pd.to_datetime(df['final_date'])

df['day_num'] = df['final_date'].dt.day

df['sentiment_category'] = np.where(df['sentiment']<-0.25, 'negative', 
                                    np.where(df['sentiment']>0.25, 'positive', 'neutro'))

df = df.drop(['sentiment_exteme'], axis=1)

df['sentiment_extreme'] = np.where(df['sentiment_category']=='neutro', np.nan, df['sentiment'])

#procesamiento hashtags 
# clean word
def clean_text(x):
    string = str(x)
    string = re.sub(r'[\W_]+', '', x) # remove anything that is not a letter or number
    string = string.strip() # remove spaces at the beginning and at the end of the string
    return  string.lower()

def clean_list(x, stopwords):
    return list(set([clean_text(unidecode(token)) for token in x if str(token)!='nan' and clean_text(unidecode(token)) not in stopwords]))

df['hashtags'] = df['hashtags'].apply(lambda x: clean_list(literal_eval(x), stopwords))

flat_list = []
for sublist in df['hashtags']:
    for item in sublist:
        flat_list.append(item)

list_hashtags = set(flat_list)
#fin preprocesamiento 


app = dash.Dash(__name__, external_stylesheets=external_stylesheets )

colors = {'text': 'grey'}

test_png = 'corebi_logo.png'
test_base64 = base64.b64encode(open(test_png, 'rb').read()).decode('ascii')

navbar = dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    html.Div(dbc.Col(html.Img(src='data:image/png;base64,{}'.format(test_base64))), 
                    style={"margin-left" : 15}),
                ],
                align="center",
                no_gutters=True,
            ),
        ),
    ],
    color="dark",
    dark=True,
)

app.layout = html.Div([
    html.Div(navbar),
    dbc.Row([
    dbc.Col([
    
    html.Div([
    dbc.Jumbotron(
            [html.H4("Bienvenido al Elector 360!", className="alert-heading", style = {"font-weight": "bold", "FontSize":30}),
             html.P('Aquí podrá hacer análsis de tweets en tiempo real', className="alert-heading"),
             html.Div(html.Hr(className="my-2"),style={"margin-bottom":3}),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader("Seleccione la fecha a analizar"), 
                            dbc.CardBody(
                                dcc.DatePickerSingle(
                                    id='my-date-picker-single',
                                    min_date_allowed=dt(1995, 8, 5),
                                    max_date_allowed=dt(2017, 9, 19),
                                    initial_visible_month=dt(2017, 8, 5),
                                    date=str(dt(2017, 8, 25, 23, 59, 59))
                                    ),
                            ),
                        ]),
                    ], style = {"margin-bottom":10})
                ] 
                    ),
            ]),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        dbc.Card([
                            dbc.CardHeader("Seleccione el tipo de TWEET a analizar"), 
                            dbc.CardBody(
                                dbc.Checklist(
                                    id = "sentimental",
                                    options=[
                                        {'label': 'Positivos', 'value': 'positive'},
                                        {'label': 'Neutros  ', 'value': 'neutro'},
                                        {'label': 'Negativos  ', 'value': 'negative'}
                                    ],
                                    value=['positive', 'neutro', 'negative'],
                                    #style ={"font-size": 20}
                                )  
                                        
                            ),
                        ]),  
                    ],style = {"margin-bottom":10})
                ], ) 
            ]),
            
            dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Seleccione el barrio de Córdoba a analizar"), 
                                    dbc.CardBody(
                                        dcc.Dropdown(
                                        id='barrio',
                                        options= [{'label': i, 'value': j} for i,j in zip(df['neighborhood'].dropna().unique(),df['neighborhood'].dropna().unique())],
                                        value=[],
                                        multi=True
                                        )
                                    ), 
                            ]), 
                        ], style = {"margin-bottom":10,'margin-right': 2}),]), 
                     ]), 
            dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("Seleccione los #HASHTAG a analizar"), 
                                dbc.CardBody(
                                    dcc.Dropdown(
                                        id='hashtag',
                                        options= [{'label': i, 'value': j} for i,j in zip(list_hashtags,list_hashtags)],
                                        value=[],
                                        multi=True
                                    ),
                                )  
                            ]),
                        ], style = {"margin-bottom":10,'margin-right': 2}),]), 

            ]),

            dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Card([
                                dbc.CardHeader("¿Desea incorporar los RETWEETS a su análisis?"), 
                                dbc.CardBody(
                                    dbc.RadioItems(
                                        id = "retweets", 
                                        options=[
                                            {'label': 'Si', 'value': 'si'},
                                            {'label': 'No', 'value': 'no'},
                                        ],
                                        value='no'
                                    )  
                                )    
                            ]),
                        ], style = {"margin-bottom":10,'margin-right': 2}),
                    ]), 

            ]),

            dbc.Row([
                 dbc.Col(  
                     html.Div(dbc.Button('Ir',  id="boton-ir", outline=True, color="dark"), 
                     style={"margin-top":4, "margin-left":7}), width={ "offset": 5}),
            ]), 
            ],style={"margin-left":15, "margin-right":7}),
    ], style ={"margin-left":10, "margin-right":6}

    #             ],style={"margin-left":15, "margin-right":700}),
    # ], style ={"margin-left":10, "margin-right":650}
),

    
    
    
    ]),

    dbc.Col([
        html.Div([
        dbc.Card([
                dbc.CardHeader("Métricas TWEETS"), 
                dbc.CardBody(
        dcc.Graph(id="fig_metrics_of_tweets", style={"width": "100%", "display": "inline-block"}, animate=None),
                )
        ])
        ], style = {"margin-right":30, "margin-top":15}),

        html.Div([
        dbc.Card([
                dbc.CardHeader("Sentimientos promedio total"), 
                dbc.CardBody(
        dcc.Graph(id="fig_gauge", style={"width": "100%", "display": "inline-block"}, animate=None),
                )
        ])
        ], style = {"margin-right":30, "margin-top":15})


    ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Card([
                        dbc.CardHeader("Tendencias de Hashtags"), 
                        dbc.CardBody(
                dcc.Graph(id="fig_tendencias", style={"width": "100%", "display": "inline-block"}, animate=None),
                        )
                ])
                ], style = {"margin-right":18, "margin-top":22, "margin-left": 15}),



        ]),
        dbc.Col([
            html.Div([
                dbc.Card([
                        dbc.CardHeader("TWEETS y localidades"), 
                        dbc.CardBody(
                dcc.Graph(id="geo_map_fig", style={"width": "100%", "display": "inline-block"}, animate=None),
                        )
                ])
                ], style = {"margin-right":30, "margin-top":15}),



        ]),
    
]) 
])

@app.callback(
    Output('fig_metrics_of_tweets', 'figure'), 
    [Input("sentimental", "value"), 
     Input("barrio", "value"), 
     Input("hashtag", "value"),
     Input("retweets", "value"), 
])

def filtros(sentimental, barrios, hashtags, retweets):
    data = df
    if len(barrios) != 0: 
        data = df[df["neighborhood"].isin(barrios)]

    if len(hashtags) != 0:
        data = data[data['hashtags'].astype(str).str.contains('|'.join(hashtags))]

    if len(sentimental) != 0:
        data = data[data['sentiment_category'].astype(str).str.contains('|'.join(sentimental))]

    if retweets == "si":
        data = data[data['retweet']==True]

    if retweets == "no":
        data = data[data['retweet']==False]

    from plots_elector_360 import metrics_of_tweet
    fig_metrics_of_tweets = metrics_of_tweet(
                        data,
                        retweet_col='retweet',
                        conversation_id_col='conversation_id',
                        id_col='id',
                        retweet_id_col='retweet_id',
                        user_id_col='user_id',
                        user_rt_id_col='user_rt_id',
                    )  .update_layout(height=300)

    return fig_metrics_of_tweets

@app.callback(
    [Output("geo_map_fig","figure"),
     Output("fig_tendencias","figure"),
     Output('fig_gauge', 'figure')],
    [Input("boton-ir", "n_clicks")],
    [State("sentimental", "value"), 
     State("barrio", "value"), 
     State("hashtag", "value"),
     State("retweets", "value"),

])

def filtros(n_clicks_ir, sentimental, barrios, hashtags, 
                retweets):

    if n_clicks_ir is None:
        raise PreventUpdate
    else:
        data = df

        if len(barrios) != 0: 
            data = df[df["neighborhood"].isin(barrios)]

        if len(hashtags) != 0:
            data = data[data['hashtags'].astype(str).str.contains('|'.join(hashtags))]

        if len(sentimental) != 0:
            data = data[data['sentiment_category'].astype(str).str.contains('|'.join(sentimental))]

        if retweets == "si":
            data = data[data['retweet']==True]

        if retweets == "no":
            data = data[data['retweet']==False]

        from plots_elector_360 import geo_map
        geo_map_fig = geo_map(
                data,
                'neighborhood',
                'tweet',
                'count',
                "barrios2.geojson",
                "Nombre",
                mapbox_accesstoken,
                fill_value=0,
                log_scale=False,
                threshold=0,
                colorscale=["#EF553B", "#FECB52", "#00CC96"],
                marker_opacity=0.5,
                lat=-31.4,
                lon=-64.2,
                mapbox_zoom=10,
                )

        from plots_elector_360 import ntrend_topics_bar
        fig_tendencias = ntrend_topics_bar(
                        data, # sampleeeeeeeeeeeeeeeeeee
                        'hashtags', # column with list of string
                        stopwords=[], # list of stopwords
                        ntop=20,
                        color_col='sentiment_extreme', # column numeric values
                        aggfunc_name='mean', # function to color column
                        fill_value=0,
                        title=None,
                        color_continuous_scale=["#EF553B", "#FECB52", "#00CC96"],
                        range_color=[-1, 1],
                        )

        fig0 = go.Indicator(
                mode = "gauge+number",
                value = data['sentiment'].mean(),
                visible = True,
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 0.1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 0.033], 'color': 'red'},
                        {'range': [0.033, 0.067], 'color': 'yellow'},{'range': [0.067, 0.1], 'color': 'green'}],
                })

        fig_gauge=go.Figure(data=fig0)
        fig_gauge.update_layout(
            autosize=True,)

    return geo_map_fig, fig_tendencias, fig_gauge

if __name__ == '__main__':
    app.run_server(debug=True)