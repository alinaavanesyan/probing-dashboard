import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import json
from glob import glob
import json
from multiprocessing import Pool, RLock
import pandas as pd
import multiprocessing as mp
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html
from statistics import mean
import math
from collections import OrderedDict
import collections
import statistics
import os
import dash_daq as daq

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

hits = glob("Probing_framework/results/*/*/*.json", recursive=True)

lang_file = pd.read_csv('all_languages.csv', delimiter=';')

with open('data/all_categories.json', 'r', encoding='utf-8') as f:
    all_categories = json.load(f)
with open('data/middle_all_layers_family.json', 'r', encoding='utf-8') as f:
    middle_all_layers_family = json.load(f)
with open('data/all_layers_lang.json', 'r', encoding='utf-8') as f:
    all_layers_lang = json.load(f)
with open('data/all_layers_lang_middle.json', 'r', encoding='utf-8') as f:
    all_layers_lang_middle = json.load(f)
with open('data/lang_files.json', 'r', encoding='utf-8') as f:
    lang_files = json.load(f)
with open('data/middle_values.json', 'r', encoding='utf-8') as f:
    middle_values = json.load(f)
with open('data/structure.json', 'r', encoding='utf-8') as f:
    structure = json.load(f)
with open('data/cat_statistics.json', 'r', encoding='utf-8') as f:
    cat_statistics = json.load(f)
with open('data/cat_statistics_for_table.json', 'r', encoding='utf-8') as f:
    cat_statistics_for_table = json.load(f)
with open('data/size.json', 'r', encoding='utf-8') as f:
    size = json.load(f)
with open('data/full_layers.json', 'r', encoding='utf-8') as f:
    full_layers = json.load(f)
with open('data/middle_values_lang.json', 'r', encoding='utf-8') as f:
    middle_values_lang = json.load(f)
with open('data/datasets.json', 'r', encoding='utf-8') as f:
    datasets = json.load(f)
with open('data/model_name.txt', 'r') as f:
    model_names = [line.rstrip() for line in f]
with open('data/middle_for_each_cat.json', 'r', encoding='utf-8') as f:
    middle_for_each_cat = json.load(f)

# This method is to calculate the distance formula between two points
def euc_dist(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))
# This is the specific process of calculating the Frechet Distance distance, which is calculated recursively
def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]
 
 # This is the method called for us
def frechet_distance(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca, len(P)-1, len(Q)-1, P, Q) # ca is the matrix of a*b (3*4), 2, 3

df_full_layers = pd.read_csv('data/df_full_layers.csv')
different_models = [{'label': str(i), 'value': str(i)} for i in model_names]
boxplot = pd.read_csv('data/boxplot.csv', delimiter=',')
genealogy = pd.read_csv('genealogy.csv', delimiter=',')
languages_for_map = [{'label': str(i), 'value': str(i)} for i in genealogy['language'].tolist()]

@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    Output(component_id='map', component_property='figure'),
    Input(component_id='language_selection_map', component_property='value'),
)
def update_output(language):
    result = []
    cards_info = {}
    if isinstance(language, list):
        for l in language:
            info = genealogy.loc[genealogy['language'].isin([l])]
            result.append([l, info['family'].iloc[0], info['subfamily'].iloc[0], info['genus'].iloc[0], info['coordinates'].iloc[0]])
            cards_info[l] = {}
            cards_info[l]['Family'] = info['family'].iloc[0]
            if info['subfamily'].iloc[0]:
                cards_info[l]['Subfamily'] = info['subfamily'].iloc[0]
            if info['genus'].iloc[0]:
                cards_info[l]['Genus'] = info['genus'].iloc[0]
    else:
        info = genealogy.loc[genealogy['language'].isin([language])]
        result = [[language, info['family'].iloc[0], info['subfamily'].iloc[0], info['genus'].iloc[0], info['coordinates'].iloc[0]]]
        cards_info[language] = {}
        cards_info[language]['Family'] = info['family'].iloc[0]
        if info['subfamily'].iloc[0]:
            cards_info[language]['Subfamily'] = info['subfamily'].iloc[0]
        if info['genus'].iloc[0]:
            cards_info[language]['Genus'] = info['genus'].iloc[0]
    df_map = pd.DataFrame(columns = ['Language', 'Family', 'Subfamily', 'Genus', 'Latitude', 'Longitude'])
    count = 0
    for res in result:
        count += 1

        temp = {'Language': res[0], 'Family': res[1], 'Subfamily': res[2], 'Genus': res[3], 'Latitude': res[4][2:-2].split(', ')[0], 'Longitude': res[4][2:-2].split(', ')[1]}
        df_temp = pd.DataFrame(temp, index=[count])
        frames = [df_map, df_temp]
        df_map = pd.concat(frames)

    df_map['Latitude'] = df_map['Latitude'].apply(pd.to_numeric)
    df_map['Longitude'] = df_map['Longitude'].apply(pd.to_numeric)
    df_map = df_map.fillna('no')

    fig = px.scatter_mapbox(df_map, lat='Latitude', lon='Longitude', hover_name='Language', hover_data=['Family', 'Subfamily', 'Genus', 'Latitude', 'Longitude'],
        color_discrete_sequence=["fuchsia"], zoom=1, height=300, center = {"lat": 39, "lon": 34})
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    cards = []
    for language in cards_info.keys():
        body_text = []
        body_text.append(html.H4(f'{language}'))
        for feature in cards_info[language].keys():
            body_text.append(f'{feature}: {cards_info[language][feature]}')
        card = dbc.Card(
            [
                dbc.CardBody(
                    dbc.Row([
                            a for a in body_text
                    ])
                ),
            ],
            body=True,
            )
        cards.append(card)
    return fig

@app.callback(
    [
        Output(component_id='middle_cat', component_property='figure'),
        Output(component_id='graph1', component_property='figure'),
    ],
    Input(component_id='model_selection', component_property='value')
)
def update_output(model_name):
    df = pd.DataFrame({'Category': middle_for_each_cat[model_name].keys(), 'Middle value': middle_for_each_cat[model_name].values()})
    fig1 = px.bar(df, x='Middle value', y='Category', orientation="h", height=750)
    fig1.update_layout(showlegend=False)
    fig1.update_layout(yaxis={'categoryorder':'total ascending'})

    df_graph2 = pd.DataFrame(list(middle_values[model_name].items()), columns = ['Family', 'Middle'])
    df_graph2['Size'] = size[model_name]['number_of_languages'].values()
    try:
        fig2 = px.scatter(df_graph2,
            x='Family',
            y='Middle', 
            size='Size'
        )
        fig2.update_layout(hovermode='x unified')
    except:
        fig2 = go.Figure()
    fig2.update_layout({
    'paper_bgcolor': 'rgba(0,0,0,0)'
    })
    return fig1, fig2

@app.callback(
    Output(component_id='note_graph1', component_property='children'),
    Input(component_id='model_selection', component_property='value'),
)
def update_output(model_name):
    res = str()
    if '[Basque]*' in middle_values[model_name].keys():
       res = '*Basque is an isolate language, it is not included in any of the language groups'
    return res

@app.callback(
    Output(component_id='card_info', component_property='children'),
    Input(component_id='model_selection', component_property='value')
)
def update_output(model_name):
    sorted_size = dict(sorted(size[model_name]['number_of_files'].items(), key=lambda item: item[1], reverse=True))
    content = [html.H4('Number of files', style={"font-weight": "bold"})]
    for key in sorted_size.keys():
        description = f"{key}: {sorted_size[key]}"
        content.append(html.P(f'{description}'))

    card_info = dbc.Card(
        [
            dbc.CardBody(content),
        ],
        body=True,
        className="border-0 bg-transparent"
    )
    return card_info

@app.callback(
    Output(component_id='heatmap1', component_property='figure'),
    Input(component_id='model_selection', component_property='value'),
)
def update_output(model_name):
    df = pd.DataFrame(columns=['Category', 'Language', 'Value'])
    for language in middle_values_lang[model_name].keys():
        for category in middle_values_lang[model_name][language]['f1'].keys():
            df_temp = {'Category': category, 'Language': language, 'Value': middle_values_lang[model_name][language]['f1'][category]}
            df_temp = pd.DataFrame([df_temp])
            df = pd.concat([df, df_temp])
    df = df.sort_values('Category', ascending=False)
    fig4 = go.Figure(
        layout=go.Layout(
            height=1150,
            width=1150,
        )
    )
    fig4.add_trace(
        go.Heatmap(
        name="Number",
        y = df['Category'].tolist(),
        x = df['Language'].tolist(),
        z = np.array(df['Value'].tolist()),
        xgap = 2,
        ygap = 2,
        colorscale="Magma"
        )
    )
    return fig4

@app.callback(
    [   
        Output(component_id='dropdown', component_property='options'),
        Output(component_id='dropdown', component_property='value'),
    ],
    Input(component_id='model_selection', component_property='value'), 
)
def update_output(model_name):
    list_of_families = list(structure[model_name].keys())
    return [{'label': str(i), 'value': str(i)} for i in list_of_families], list_of_families[0]

@app.callback(
    [   
        Output(component_id='family_selection', component_property='options'),
        Output(component_id='family_selection', component_property='value'),
    ],
    Input(component_id='model_selection', component_property='value'), 
)
def update_output(model_name):
    list_of_families = list(structure[model_name].keys())
    if '[Basque]*' in list_of_families:
        list_of_families.remove('[Basque]*')
    return [{'label': str(i), 'value': str(i)} for i in sorted(list_of_families)], sorted(list_of_families)[0]

@app.callback(
    Output(component_id='graph2', component_property='figure'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='dropdown', component_property='value')
    ],
)
def update_output(model_name, families):
    df_graph2 = pd.DataFrame(columns = ['Language', 'X', 'Y'])
    if isinstance(families, list):
        for family in families:
            layers = middle_all_layers_family[model_name][family]['f1']
            x = list(layers.keys())
            y = list(layers.values())
            name = [family] * len(x)
            df = pd.DataFrame(columns = ['Language', 'X', 'Y'])
            df['Language'] = name
            df['X'] = x
            df['Y'] = y
            frames = [df_graph2, df]
            df_graph2 = pd.concat(frames)
    else:
        layers = middle_all_layers_family[model_name][families]['f1']
        x = list(layers.keys())
        y = list(layers.values())
        name = [families] * len(x)
        df = pd.DataFrame(columns = ['Language', 'X', 'Y'])
        df['Language'] = name
        df['X'] = x
        df['Y'] = y
        frames = [df_graph2, df]
        df_graph2 = pd.concat(frames)
    try:
        fig2 = px.line(df_graph2, x = 'X', y = 'Y', color='Language', labels={'X':'Layer number', 'Y': 'Value'})
    except:
        fig2 = go.Figure()
    fig2.update_layout({
    'paper_bgcolor': 'rgba(0,0,0,0)'
    })
    return fig2

@app.callback(
    [   
        Output(component_id='languages', component_property='options'),
        Output(component_id='languages', component_property='value'),
    ],
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='category', component_property='value')
    ],
)
def update_output(model_name, category):
    df_temp = df_full_layers[df_full_layers['Model'].isin([model_name])]
    df_temp = df_temp[df_temp['Category'].isin([category])]
    df_temp = df_temp[df_temp['Metric'].isin(['f1'])]
    list_of_languages = df_temp['Language'].unique().tolist()
    return [{'label': str(i), 'value': str(i)} for i in list_of_languages], list_of_languages[0]

@app.callback(
    Output(component_id='graph3', component_property='figure'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='category', component_property='value'), 
        Input(component_id='languages', component_property='value')
    ],
)
def update_output(model_name, category, languages):
    df_graph3 = pd.DataFrame(columns = ['Language', 'X', 'Y'])
    if isinstance(languages, list):
        count = -1
        for language in languages:
            count += 1
            df_temp = df_full_layers[df_full_layers['Model'].isin([model_name])]
            df_temp = df_temp[df_temp['Language'].isin([language])]  
            df_temp = df_temp[df_temp['Metric'].isin(['f1'])]
            df_temp = df_temp[df_temp['Category'].isin([category])]   
            df_temp.pop('Model'); df_temp.pop('Metric')                    
            frames = [df_temp, df_graph3]
            df_graph3 = pd.concat(frames)
    else:
        df_temp = df_full_layers[df_full_layers['Model'].isin([model_name])]
        df_temp = df_temp[df_temp['Language'].isin([languages])]     
        df_temp = df_temp[df_temp['Metric'].isin(['f1'])]
        df_temp = df_temp[df_temp['Category'].isin([category])]
        df_temp.pop('Model'); df_temp.pop('Metric')   
        frames = [df_temp, df_graph3]
        df_graph3 = pd.concat(frames)
    
    try:
        fig3 = px.line(df_graph3, x = 'X', y = 'Y', color='Language',labels={'X':'Layer number', 'Y': 'Value'})
    except:
        fig3 = go.Figure()
    fig3.update_layout({'paper_bgcolor': 'rgba(0,0,0,0)'
    })
    return fig3

@app.callback(
    [   
        Output(component_id='language_graph4', component_property='options'),
        Output(component_id='language_graph4', component_property='value'),
    ],
    Input(component_id='model_selection', component_property='value'), 
)
def update_output(model_name):
    list_of_languages = list(all_layers_lang[model_name].keys())
    return [{'label': str(i), 'value': str(i)} for i in list_of_languages], list_of_languages[0]


@app.callback(
    [   
        Output(component_id='categories_graph4', component_property='options'),
        Output(component_id='categories_graph4', component_property='value'),
    ],
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='language_graph4', component_property='value')
    ],
)
def update_output(model_name, language):
    df_temp = df_full_layers[df_full_layers['Model'].isin([model_name])]
    df_temp = df_temp[df_temp['Language'].isin([language])]
    df_temp = df_temp[df_temp['Metric'].isin(['f1'])]
    list_of_categories = df_temp['Category'].unique().tolist()
    return [{'label': str(i), 'value': str(i)} for i in list_of_categories], list_of_categories[0]

@app.callback(
    Output(component_id='graph4', component_property='figure'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='language_graph4', component_property='value'),
        Input(component_id='categories_graph4', component_property='value'), 
    ],
)
def update_output(model_name, language, categories):
    df_graph4 = pd.DataFrame(columns = ['Language', 'x', 'y'])    
    df_temp = df_full_layers[df_full_layers['Model'].isin([model_name])]
    df_temp = df_temp[df_temp['Language'].isin([language])]     
    df_temp = df_temp[df_temp['Metric'].isin(['f1'])]

    if isinstance(categories, list):
        for category in categories:
            df_temp2 = df_temp[df_temp['Category'].isin([category])]
            df_temp2.pop('Model'); df_temp2.pop('Metric')   
            frames = [df_temp2, df_graph4]
            df_graph4 = pd.concat(frames)
    else:
        df_temp2 = df_temp[df_temp['Category'].isin([categories])]
        df_temp2.pop('Model'); df_temp2.pop('Metric')   
        frames = [df_temp2, df_graph4]
        df_graph4 = pd.concat(frames)
    
    if df_graph4.empty:
        fig4 = go.Figure()
    else:
        fig4 = px.line(df_graph4, x = 'X', y = 'Y', color='Category', labels={'X':'Layer number', 'Y': 'Value'})
    fig4.update_layout({'paper_bgcolor': 'rgba(0,0,0,0)'})
    return fig4

@app.callback(
    Output(component_id='datasets', component_property='children'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='language_graph4', component_property='value'),
        Input(component_id='categories_graph4', component_property='value'), 
    ],
)
def update_output(model_name, language, categories):
    cards = []
    if isinstance(categories, list):
        for category in categories:
            body_text = []  
            body_text.append(html.H4(f'{category}', style={'font-weight': 'bold'})),
            training = [html.B("Training:")]
            for key in datasets[model_name][language][category]['training'].keys():
                training.append(html.P(f"{key} — {datasets[model_name][language][category]['training'][key]}"))
            validation = [html.B("Validation:")]
            for key in datasets[model_name][language][category]['validation'].keys():
                validation.append(html.P(f"{key} — {datasets[model_name][language][category]['validation'][key]}"))
            test = [html.B("Test:")]
            for key in datasets[model_name][language][category]['test'].keys():
                test.append(html.P(f"{key} — {datasets[model_name][language][category]['test'][key]}"))
            
            body_text.append(dbc.Col([
                                    a for a in training
                                    ])
            )
            body_text.append(dbc.Col([
                                    a for a in validation
                                    ])
            )
            body_text.append(dbc.Col([
                                    a for a in test
                                    ])
            )

            card = dbc.Card(
                [
                    dbc.CardBody(
                        dbc.Row([
                                a for a in body_text
                        ])
                    ),
                ],
                className='bg-transparent',
                style={'color': 'black'},
                body=True,
            )
            cards.append(card)

    else:
        category = categories
        body_text = []  
        body_text.append(html.H4(f'{category}', style={'font-weight': 'bold'}))
        training = [html.B("Training:")]
        for key in datasets[model_name][language][category]['training'].keys():
            training.append(html.P(f"{key} — {datasets[model_name][language][category]['training'][key]}"))
        validation = [html.B("Validation:")]
        for key in datasets[model_name][language][category]['validation'].keys():
            validation.append(html.P(f"{key} — {datasets[model_name][language][category]['validation'][key]}"))
        test = [html.B("Test:")]
        for key in datasets[model_name][language][category]['test'].keys():
            test.append(html.P(f"{key} — {datasets[model_name][language][category]['test'][key]}"))
        
        body_text.append(dbc.Col([
                                a for a in training
                                ])
        )
        body_text.append(dbc.Col([
                                a for a in validation
                                ])
        )
        body_text.append(dbc.Col([
                                a for a in test
                                ])
        )

        card = dbc.Card(
            [   
                dbc.CardBody(
                    dbc.Row([
                            a for a in body_text
                    ])
                ),
            ],
            className='bg-transparent',
            style={'color': 'black'},
            body=True,
        )
        cards.append(card)
    
    children = [
            dbc.Row([
                a for a in cards
            ])
    ]
    return children

@app.callback(
    Output(component_id='card1', component_property='children'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='family_selection', component_property='value')
    ]
)
def update_output(model_name, family):
    languages = []
    for language in structure[model_name][family]:
        try:
            code = lang_file.loc[lang_file['Language'].isin([language])]
            code = code.iloc[0]['Codes']
            
            count = len(lang_files[model_name][family][code])
            if count > 0:
                languages.append(language)
        except:
            value = f'The language {language} is part of the language family, but is not represented in the files with the results of probing'
    card1 = dbc.Card(
        [
            dbc.CardBody(
            [
                html.H2(f'{len(languages)}'),
                html.P('Number of languages in this language family', className='card-text'),
            ]
            ),
        ],
        body=True,
        color='#2D2D2D',
        inverse=True,
    )
    return card1

@app.callback(
    Output('card2', 'children'),
    Input('model_selection', 'value'), Input('family_selection', 'value')
)
def update_output(model_name, family):
    min_count = 0
    min_count = min(all_categories[model_name][family]['f1'].items(), key=lambda x: x[1])
    card2 = dbc.Card(
        [                       
            dbc.CardBody(
            [
                html.H2(f'{str(min_count[0])}'),
                html.P('The most poorly recognized category by the model', className='card-text'),      
            ]
            ),
        ],
        body=True,
        color='#2D2D2D',
        inverse=True,
    )
    return card2

@app.callback(
    Output(component_id='card3', component_property='children'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='family_selection', component_property='value')
    ]
)
def update_output(model_name, family):
    max_count = 0
    max_count = max(all_categories[model_name][family]['f1'].items(), key=lambda x: x[1])
    card3 = dbc.Card(
        [    
            dbc.CardBody(
            [
                html.H2(f'{str(max_count[0])}'),
                html.P('The most well recognized category by the model', className='card-text'),
            ]
            ),
        ],
        body=True,
        color='#2D2D2D',
        inverse=True,
    )
    return card3

@app.callback(
    Output('treemap1', 'figure'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='family_selection', component_property='value')
    ]
)
def update_output(model_name, family):
    labels = []
    labels.append(family)
    parents = []
    parents.append('')
    values = []
    values.append(0)
    for language in structure[model_name][family]:
        try:
            labels.append(language)
            parents.append(family)

            code = lang_file.loc[lang_file['Language'].isin([language])]
            code = code.iloc[0]['Codes']
            
            count = len(lang_files[model_name][family][code])
            values.append(count)
        except:
            value = f'The language {language} is part of the language family, but is not represented in the files with the results of probing'
    values[0] = sum(values)
    treemap1 = go.Figure(go.Treemap(
        labels = labels,
        parents= parents,
        values= values,
        root = None)
    )
    treemap1.update_layout(
        font_size=20,
        margin = dict(t=20, l=15, r=20, b=20),
    )
    treemap1.update_layout({'paper_bgcolor': 'rgba(0,0,0,0)'})
    return treemap1

@app.callback(
    Output('boxplot', 'figure'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='family_selection', component_property='value'),
        Input(component_id='tick', component_property='on'),
    ]
)

def update_output(model_name, user_family, tick):
    boxplot_for_family = boxplot[boxplot['Model name'].isin([model_name])]
    boxplot_for_family = boxplot_for_family[boxplot_for_family['Family'].isin([user_family])]
    x = "{}".format(tick)
    if x == 'True' :
        tr1 = px.box(boxplot_for_family, x= 'Category', y='Average value')
        tr2 = px.scatter(boxplot_for_family, x='Category', y='Average value', color='Language')
        fig = go.Figure(data=tr1.data + tr2.data)
    else:
        tr1 = px.box(boxplot_for_family, x= 'Category', y='Average value')
        fig = go.Figure(tr1.data)
    fig.update_layout({'paper_bgcolor': 'rgba(0,0,0,0)'})
    return fig

@app.callback(
    [   
        Output(component_id='quantity_of_languages', component_property='options'),
        Output(component_id='quantity_of_languages', component_property='value'),
        Output(component_id='quantity_of_languages', component_property='style'),
    ],
    [   
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='family_selection', component_property='value')
    ],
)
def update_output(model_name, family):
    family_structure = structure[model_name][family]
    numbers = []
    df_spec = pd.DataFrame(columns = ['Language', 'Category', 'X', 'Y'])
    for language in family_structure:
        df_temp = df_full_layers[df_full_layers['Model'].isin([model_name])]
        df_temp = df_temp[df_temp['Language'].isin([language])]
        df_temp = df_temp[df_temp['Metric'].isin(['f1'])]
        df_temp.pop('Model'); df_temp.pop('Metric')  
        frames = [df_spec, df_temp]
        df_spec = pd.concat(frames)
    flag = 0
    for category in df_spec['Category'].unique().tolist():
        check = df_spec.loc[df_spec['Category'].isin([category])]
        numbers.append(len(check['Language'].unique().tolist()))
        if len(check['Language'].unique().tolist()) > 3:
            flag = 1
    if len(family_structure) > 1:
        quantity = [i for i in range(1, max(numbers)+1)]
    else:
        quantity = [1]
    if flag == 0:
        style={'display': 'none'}
    else:
        style={'visibility':'visible'}
    return [{'label': int(i), 'value': int(i)} for i in quantity], quantity[0], style

@app.callback(
    Output('graphs_for_family', 'children'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='family_selection', component_property='value'), 
        Input(component_id='quantity_of_languages', component_property='value')
    ],
)
def update_output(model_name, family, quantity):
    df_similar_graph1 = pd.DataFrame(columns = ['Language', 'Category', 'X', 'Y'])
    df_dissimilar_graph2 = pd.DataFrame(columns = ['Language', 'Category', 'X', 'Y'])
    model_name_new =  model_name.replace('/', '%')

    if family != '[Basque]*':
        path = f'data/graphs/{model_name_new}/{family}'
    else:
        path = f'data/graphs/{model_name_new}/{family[:-1]}'
    df_info_for_graphs = pd.read_csv(f'{path}/df_info_for_graphs.csv')
    for category in df_info_for_graphs['Category'].unique().tolist():
        path_with_cat = f'{path}/{category}'
        if os.path.exists(f'{path_with_cat}/distances1.json'):
            with open(f'{path_with_cat}/distances1.json', 'r', encoding='utf-8') as f:
                distances1 = json.load(f)
            with open(f'{path_with_cat}/distances2.json', 'r', encoding='utf-8') as f:
                distances2 = json.load(f)
            
            distances1_names = list(distances1.keys())
            distances2_names = list(distances2.keys())            
            if quantity:
                if quantity > 1:
                    for b in range(quantity):
                        if b >= len(distances1_names):
                            break

                        df_temp = df_info_for_graphs[df_info_for_graphs['Language'].isin([distances1_names[b]])]
                        df_temp = df_temp.loc[df_temp['Category'].isin([category])]
                        frames = [df_similar_graph1, df_temp]
                        df_similar_graph1 = pd.concat(frames)

                        df_temp = df_info_for_graphs[df_info_for_graphs['Language'].isin([distances2_names[b]])]
                        df_temp = df_temp.loc[df_temp['Category'].isin([category])]
                        frames = [df_dissimilar_graph2, df_temp]
                        df_dissimilar_graph2 = pd.concat(frames)
                else:
                    df_temp = df_info_for_graphs[df_info_for_graphs['Language'].isin([distances1_names[0]])]
                    df_temp = df_temp.loc[df_temp['Category'].isin([category])]
                    frames = [df_similar_graph1, df_temp]
                    df_similar_graph1 = pd.concat(frames)

                    df_temp = df_info_for_graphs[df_info_for_graphs['Language'].isin([distances2_names[0]])]
                    df_temp = df_temp.loc[df_temp['Category'].isin([category])]
                    frames = [df_dissimilar_graph2, df_temp]
                    df_dissimilar_graph2 = pd.concat(frames)
        if os.path.exists(f'{path_with_cat}/df_incomparable_graph3.csv'):
            df_incomparable_graph3 = pd.read_csv(f'{path_with_cat}/df_incomparable_graph3.csv')    

    graphs_for_div = []
    similar_graphs_for_col = []
    dissimilar_graphs_for_col = []
    incomparable_graphs_for_col = []
    if df_similar_graph1.empty == False:
        for category in df_similar_graph1['Category'].unique().tolist():
            df_temp = df_similar_graph1[df_similar_graph1['Category'].isin([category])]
            fig1 = px.line(df_temp, x='X', y='Y', color='Language', 
                        labels={'X':'Layer number', 'Y': 'Value'}, title=f'<b>{category}</b>')
            fig1['layout'].update(height=350)
            fig1['layout'].update(yaxis_range=[0,1])
            fig1.update_layout({
                'paper_bgcolor': 'rgba(0,0,0,0)'
            })
            row = dcc.Graph(figure=fig1)
            if len(similar_graphs_for_col) == 0:
                similar_graphs_for_col.append(html.H5('The most similar trends by category:',  style={'font-weight': 'bold', 'text-align': 'center'}))
            similar_graphs_for_col.append(row)
    if df_dissimilar_graph2.empty == False:
        for category in df_dissimilar_graph2['Category'].unique().tolist():
            df_temp = df_dissimilar_graph2[df_dissimilar_graph2['Category'].isin([category])]
            fig2 = px.line(df_temp, x='X', y='Y', color='Language', 
                        labels={'X':'Layer number', 'Y': 'Value'}, title=f'<b>{category}</b>')
            fig2['layout'].update(height=350)
            fig2['layout'].update(yaxis_range=[0,1])
            fig2.update_layout({
                'paper_bgcolor': 'rgba(0,0,0,0)'
            })
            row = dcc.Graph(figure=fig2)
            if len(dissimilar_graphs_for_col) == 0:
                dissimilar_graphs_for_col.append(html.H5('The most dissimilar trends by category:', style={'font-weight': 'bold', 'text-align': 'center'}))
            dissimilar_graphs_for_col.append(row)
    if df_incomparable_graph3.empty == False:
        for category in df_incomparable_graph3['Category'].unique().tolist():
            df_temp = df_incomparable_graph3[df_incomparable_graph3['Category'].isin([category])]
            fig3 = px.line(df_temp, x='X', y='Y', color='Language', 
                        labels={'X':'Layer number', 'Y': 'Value'}, title=f'<b>{category}</b>')
            fig3['layout'].update(height=350)
            fig3['layout'].update(yaxis_range=[0,1])
            fig3.update_layout({
                'paper_bgcolor': 'rgba(0,0,0,0)'
            })
            row = dcc.Graph(figure=fig3)
            if len(incomparable_graphs_for_col) == 0:
                incomparable_graphs_for_col.append(html.H5('Comparison is impossible, the number of languages is too small',  style={'font-weight': 'bold', 'text-align': 'center'}))
            incomparable_graphs_for_col.append(row)

    if len(similar_graphs_for_col) != 0:
        graphs_for_div.append(dbc.Col([a for a in similar_graphs_for_col], width=4))
    if len(dissimilar_graphs_for_col) != 0:
        graphs_for_div.append(dbc.Col([a for a in dissimilar_graphs_for_col], width=4))
    if len(incomparable_graphs_for_col) != 0:
        graphs_for_div.append(dbc.Col([a for a in incomparable_graphs_for_col], width=4))

    children = [
                dbc.Row([
                    a for a in graphs_for_div
                ])
                ]
    graphs = html.Div(children=children)
    return graphs

@app.callback(
    [   Output(component_id='category', component_property='options'),
        Output(component_id='category', component_property='value'),
    ],
    Input(component_id='model_selection', component_property='value')
)
def update_output(model_name):
    all_cats = []
    for family in all_categories[model_name].keys():
        for category in all_categories[model_name][family]['f1'].keys():
            if category not in all_cats:
                all_cats.append(category)
    return [{'label': str(i), 'value': str(i)} for i in all_cats], all_cats[0]

@app.callback(
    [   Output(component_id='category_heatmap2', component_property='options'),
        Output(component_id='category_heatmap2', component_property='value'),
    ],
    Input(component_id='model_selection', component_property='value'), 
)
def update_output(model_name):
    df_temp = df_full_layers[df_full_layers['Model'].isin([model_name])]
    df_temp = df_temp[df_temp['Metric'].isin(['f1'])]
    list_of_categories = df_temp['Category'].unique().tolist()
    return [{'label': str(i), 'value': str(i)} for i in list_of_categories], list_of_categories[0]

@app.callback(
    Output(component_id='heatmap2', component_property='figure'),
    [
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='category_heatmap2', component_property='value'),
    ]
)
def update_output(model_name, category):
    df_heatmap = df_full_layers[df_full_layers['Model'].isin([model_name])]
    df_heatmap = df_heatmap[df_heatmap['Category'].isin([category])]
    df_heatmap = df_heatmap[df_heatmap['Metric'].isin(['f1'])]
    df_heatmap = df_heatmap.sort_values('Language', ascending=False)
    fig5 = go.Figure(
        layout=go.Layout(
        height=1000,
        )
    )
    if category != '':
        fig5.add_trace(
            go.Heatmap(
            name="Number",
            y = df_heatmap['Language'].tolist(),
            x = df_heatmap['X'].tolist(),
            z = np.array(df_heatmap['Y'].tolist()),
            xgap = 2,
            ygap = 2,
            colorscale="Magma"
            )
        )
    return fig5


if __name__ == "__main__":  
    app.layout = html.Div(children=[
        dbc.Row([
                html.Div(children=[
                    dbc.Row([
                        dbc.Col([
                            html.H1('Probing results', style={'font-weight': 'bold'}),
                        ], width=7),

                        dbc.Col([
                            html.Div([
                                html.H5('Choose a model:', style={'margin-right': '10px'}),
                                html.Div([
                                    dcc.Dropdown(
                                        id='model_selection',
                                        options = different_models,
                                        value = model_names[0], 
                                    ),
                                ], style={'width': '55%', 'margin-right': '1rem'}),
                                html.Div([
                                    dbc.Button('Manual', id='open', n_clicks=0, color='secondary'),
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader(dbc.ModalTitle('Manual')),
                                            dbc.ModalBody('Here is the manual'),
                                            dbc.ModalFooter(
                                                dbc.Button(
                                                    'Close', id='close', className='ms-auto', n_clicks=0,
                                                )
                                            ),
                                        ],
                                        id='modal',
                                        is_open=False,
                                    ),
                                ]),
                            ], style={'display': 'flex', 'flex-direction': 'row'})
                        ], width=5),

                    ]),
                ]),        
                html.Hr(),
                dbc.Card([
                    dbc.CardBody(
                        dbc.Row([
                            dbc.Col([
                                html.H3('Language map'),
                                html.P('Description'),
                                dcc.Dropdown(
                                    id='language_selection_map',
                                    options = languages_for_map,
                                    value = 'Abun', 
                                    multi=True
                                ),
                            ], width=5),

                            dbc.Col([
                                dcc.Graph(id='map'),
                            ], width=7),
                        ]),  
                    ),
                ],
                    body=True,
                    color='#2D2D2D',
                    inverse=True
                ),
                html.Div(children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                    html.H4('Average value for each category:', style={'font-weight': 'bold'}, className='text-center'),
                                    html.P('graph 1', style={'font-style': 'italic'}, className='text-center'),
                                    dcc.Graph(id='middle_cat'),
                            ]),
                        ], width=3, style={'border-right': '1px solid', 'border-color': '#cfcfcf'}),

                        dbc.Col([
                            html.Div([
                                    html.H4('Average values for all families, layers and categories:', style={'font-weight': 'bold'}, className='text-center'),
                                    html.P('graph 2', style={'font-style': 'italic'}, className='text-center'),
                                    dcc.Graph(id='graph1'),
                                    html.P(id='note_graph1', style={'color': '#999999', 'font-style': 'italic', 'margin-top': '1.5rem'}, className='text-center')
                            ]),
                        ], width=6, style={'border-right': '1px solid', 'border-color': '#cfcfcf'}),

                        dbc.Col([
                            html.Div(id='card_info'),
                        ], width=2)     

                    ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'}),
                ]),
                html.Br(),
                html.Div(children=[
                    dbc.Row([
                        dbc.Col([
                            html.H4('Average values for all categories\n(for each language family):', style={'text-align': 'center', 'font-weight': 'bold'}),
                            html.P('graph 3', style={'font-style': 'italic'}, className='text-center'),
                            dcc.Dropdown(
                                id='dropdown',  
                                multi=True
                            ),
                            html.Div(dcc.Graph(id='graph2')),
                        ], width=6),
                
                        dbc.Col([
                                html.H4('Values for each layer and category (for each language):', style={'text-align': 'center', 'font-weight': 'bold'}),
                                html.P('graph 4', style={'font-style': 'italic'}, className='text-center'),
                                dcc.Dropdown(id='category'),
                                dcc.Dropdown(id='languages', multi=True, style={"margin-top": "0.5rem"},), 
                                dcc.Graph(id='graph3'),
                        ], width=6),
                    ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '3rem', 'border-radius': '4px', 'margin-top': '2em', 'background-color': '#F5F9FF'})
                                    
                ]),
                html.Div(children=[
                    dbc.Row([
                        dbc.Col([
                                html.H4('Values for each layer and category (for certain language):', style={'text-align': 'center', 'font-weight': 'bold'}),
                                html.P('graph 5', style={'font-style': 'italic'}, className='text-center'),
                                dcc.Dropdown(id='language_graph4'), 
                                dcc.Dropdown(id='categories_graph4', multi=True, style={"margin-top": "0.5rem"}),
                                dcc.Graph(id='graph4'),
                        ], width=7, style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'}),
                        dbc.Col([
                            # html.Div(id='datasets', style={"maxHeight": "550px", "overflow": "scroll", "background-color": "linear-gradient(to bottom, green 25%, #FFFFFF 0%)"}, className="border-1 bg-purple-gradient",),
                            html.Div(id='datasets', className='bg-transparent', style={'maxHeight': '550px', 'overflow': 'scroll', 'background-color': 'linear-gradient(to bottom, green 25%, #FFFFFF 0%)'}),
                        ], width=4, className='offset-md-1', style={'color': 'white', 'background': 'linear-gradient(#D4C0FF, #C2DAFD)', 'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'})
                    ],)
                                    
                ]),
                html.Div(children=[
                    html.H3('Statistics on language families', style={'text-align': 'center', 'padding-top': '3rem', 'font-weight': 'bold'}),
                    html.Br(),
                    dcc.Dropdown(id='family_selection'),

                    dbc.Row([
                        dbc.Col(id='card1', width=4),
                        dbc.Col(id='card2', width=4),
                        dbc.Col(id='card3', width=4),
                        ], style={'padding-top': '2rem', 'padding-bottom': '2rem'}),
                        
                    html.Div(children=[
                            dbc.Row([
                                dbc.Col([
                                        html.H4('Structure of the language family:', style={'text-align': 'center', 'font-weight': 'bold'}, className='text-center'),
                                        html.P('graph 6', style={'font-style': 'italic'}, className='text-center'),
                                        dcc.Graph(id='treemap1'),
                                        ], width=4),
                                dbc.Col([
                                        html.H4('Average values by category:', style={'text-align': 'center', 'font-weight': 'bold'}),
                                        html.P('graph 7', style={'font-style': 'italic'}, className='text-center'),
                                        

                                        html.Div(className='box',
                                            children=[html.Div([
                                                html.Tr([
                                                    html.Td(daq.BooleanSwitch(id='tick', on=False, color='#9B51E0')),
                                                    html.Td(html.P('Show languages', id='tick_lable')),
                                                ]),
                                                dcc.Graph(id='boxplot'),
                                            ])
                                        ]),                             
                                        
                    ], width=8),

                                ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'})
                            ], ),
    
                ]),
                dbc.Row([
                        html.P('graph 8', style={'font-style': 'italic'}, className='text-center'),
                        dbc.Col(dcc.Dropdown(id='quantity_of_languages'), width=8, style={'margin-bottom': '2rem', 'width': '30%'}),
                        html.Div(id='graphs_for_family'),
                        ], style={'margin-bottom': '3em', 'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'margin-bottom': '2rem', 'background-color': '#ffffff'}),
                
                html.Div(children=[
                    dbc.Row([
                        html.H4('Average values for all layers (for each "category-language" pair):', style={'text-align': 'center', 'font-weight': 'bold'}, className='text-center'),
                        html.P('graph 9', style={'font-style': 'italic'}, className='text-center'),
                        dcc.Graph(id='heatmap1'),
                    ], justify='center')
                ], style={'align': 'center', 'justify': 'center', 'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'padding-bottom': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'}),

                html.Div(children=[
                    html.H4('Values for each layer for each language and each category:', style={'text-align': 'center', 'font-weight': 'bold'}, className='text-center'),
                    html.P('graph 10', style={'font-style': 'italic'}, className='text-center'),
                    dcc.Dropdown(id='category_heatmap2'),
                    dcc.Graph(id='heatmap2')
                ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'}),

                ], justify='center')
                        
        ], style={'padding': '3rem'})

    app.run_server(debug=True)