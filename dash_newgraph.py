import collections
import json
import math
import multiprocessing as mp
import os
import statistics
from collections import OrderedDict
from glob import glob
from multiprocessing import Pool, RLock
from statistics import mean

import dash
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import seaborn as sn
from dash import Dash, Input, Output, dcc, html
from plotly.subplots import make_subplots

PORT = os.getenv("APP_DASHBOARD_PORT")
try:
    PORT = int(PORT)
except:
    PORT = 8050

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], url_base_pathname="/dashboard")

server = app.server

hits = glob("Probing_framework4/results/*/*/*.json", recursive=True)

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

with open('data/model_name.txt', 'r') as f:
    model_names = [line.rstrip() for line in f]

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

@app.callback(
    Output(component_id='graph1', component_property='figure'),
    Input(component_id='model_selection', component_property='value')
)
def update_output(model_name):
    df_graph1 = pd.DataFrame(list(middle_values[model_name].items()), columns = ['Family', 'Middle'])
    df_graph1['Size'] = size[model_name]['number_of_languages'].values()
    try:
        fig1 = px.scatter(df_graph1,
            x='Family',
            y='Middle', 
            size='Size'
        )
        fig1.update_layout(hovermode='x unified')
    except:
        fig1 = go.Figure()
    return fig1

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

    content = [html.H2('Number of files')]
    for key in sorted_size.keys():
        description = f"{key}: {sorted_size[key]}"
        content.append(html.P(f'{description}'))

    card_info = dbc.Card(
        [
            dbc.CardBody(
                content
            ),
        ],
        body=True,
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
    [   Output(component_id='dropdown', component_property='options'),
        Output(component_id='dropdown', component_property='value'),
    ],
    Input(component_id='model_selection', component_property='value'), 
)
def update_output(model_name):
    list_of_families = list(structure[model_name].keys())
    return [{'label': str(i), 'value': str(i)} for i in list_of_families], list_of_families[0]

@app.callback(
    [   Output(component_id='family_selection', component_property='options'),
        Output(component_id='family_selection', component_property='value'),
    ],
    Input(component_id='model_selection', component_property='value'), 
)
def update_output(model_name):
    list_of_families = list(structure[model_name].keys())
    if '[Basque]*' in list_of_families:
        list_of_families.remove('[Basque]*')
    return [{'label': str(i), 'value': str(i)} for i in list_of_families], list_of_families[0]

@app.callback(
    Output(component_id='graph2', component_property='figure'),
    [   Input(component_id='model_selection', component_property='value'), 
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
    return fig2

@app.callback(
    [   Output(component_id='languages', component_property='options'),
        Output(component_id='languages', component_property='value'),
    ],
    [   Input(component_id='model_selection', component_property='value'), 
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
    [   Input(component_id='model_selection', component_property='value'), 
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
    
    return fig3


@app.callback(
    [   Output(component_id='language_graph4', component_property='options'),
        Output(component_id='language_graph4', component_property='value'),
    ],
    Input(component_id='model_selection', component_property='value'), 
)
def update_output(model_name):
    list_of_languages = list(all_layers_lang[model_name].keys())
    return [{'label': str(i), 'value': str(i)} for i in list_of_languages], list_of_languages[0]


@app.callback(
    [   Output(component_id='categories_graph4', component_property='options'),
        Output(component_id='categories_graph4', component_property='value'),
    ],
    [   Input(component_id='model_selection', component_property='value'), 
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
    [   Input(component_id='model_selection', component_property='value'), 
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

    return fig4


@app.callback(
    Output(component_id='card1', component_property='children'),
    [   Input(component_id='model_selection', component_property='value'), 
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
                html.P('Number of languages in a given language family', className='card-text'),
            ]
            ),
        ],
        body=True,
        color="primary",
        inverse=True,
    )
    return card1

@app.callback(
    Output('card2', 'children'),
    [Input('model_selection', 'value'), Input('family_selection', 'value')]
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
        color="primary",
        inverse=True,
    )
    return card2

@app.callback(
    Output(component_id='card3', component_property='children'),
    [   Input(component_id='model_selection', component_property='value'), 
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
        color="primary",
        inverse=True,
    )
    return card3

@app.callback(
    Output('treemap1', 'figure'),
    [   Input(component_id='model_selection', component_property='value'), 
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
        margin = dict(t=20, l=15, r=20, b=20)
    )
    return treemap1

@app.callback(
    Output('table1', 'figure'),
    [   Input(component_id='model_selection', component_property='value'), 
        Input(component_id='family_selection', component_property='value'),
    ]
)
def update_output(model_name, user_family):
    df_table1 = pd.DataFrame(columns = ['Language family', 'Category', 'Minimum value', 'Maximum value'])
    for family in cat_statistics_for_table[model_name].keys():
        if family == user_family:
            for category in cat_statistics_for_table[model_name][family].keys():
                if isinstance(cat_statistics_for_table[model_name][family][category], list):
                    row = [family, category, cat_statistics_for_table[model_name][family][category][0], cat_statistics_for_table[model_name][family][category][1]]
                    df_table1.loc[len(df_table1.index)] = row
                else:
                    row = [family, category, cat_statistics_for_table[model_name][family][category], cat_statistics_for_table[model_name][family][category]]
                    df_table1.loc[len(df_table1.index)] = row
    
    column1 = df_table1['Category']
    column2 = df_table1['Minimum value']
    column3 = df_table1['Maximum value']

    table1 = go.Figure(data=[go.Table(header=dict(values=['Category', 'Minimum value', 'Maximum value']),
                    cells=dict(values=[column1, column2, column3]))
                    ])
    
    table1.update_layout(margin=dict(l=20, r=20, t=20, b=20),)
    return table1

@app.callback(
    [   Output(component_id='quantity_of_languages', component_property='options'),
        Output(component_id='quantity_of_languages', component_property='value'),
        Output(component_id='quantity_of_languages', component_property='style'),
    ],
    [   Input(component_id='model_selection', component_property='value'),
        Input(component_id='family_selection', component_property='value')
    ],
)
def update_output(model_name, family):
    family_structure = structure[model_name][family]
    if len(family_structure) > 1:
        quantity = [i for i in range(1, len(family_structure))]
    else:
        quantity = [1]
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
        if len(check['Language'].unique().tolist()) > 3:
            flag = 1
            break
    if flag == 0:
        style={'display': 'none'}
    else:
        style={'visibility':'visible'}
    return [{'label': int(i), 'value': int(i)} for i in quantity], quantity[0], style

@app.callback(
    Output('graphs_for_family', 'children'),
    [   Input(component_id='model_selection', component_property='value'), 
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
                        html.H1('Probing results'),
                    ], width=8),

                    dbc.Col([
                        html.H5('Choose a model:'),
                    ], width=2),

                    dbc.Col([
                        dcc.Dropdown(
                            id='model_selection',
                            options = different_models,
                            value = model_names[0], 
                            style={'width': '100%' }),
                    ], width=2),
                ]),
            ]),
                
            html.Hr(),
            html.Div(children=[
                dbc.Row([
                    dbc.Col([
                        html.H5('Average values for all families, layers and categories:', style={"padding-top": "2rem"}),
                        dcc.Graph(id='graph1'),
                        html.P(id='note_graph1', style={"padding-bottom": "2rem"})
                    ], width=8),
                    dbc.Col(id='card_info', width=4),
                ]),
            ]),
            html.Br(),
            html.Div(children=[
                dbc.Row([
                    dbc.Col([
                        html.H5('Average values for all categories (for each language family):', style={'text-align': 'center'}),
                        dcc.Dropdown(
                            id='dropdown',  
                            multi=True
                        ),
                        html.Div(dcc.Graph(id='graph2')),
                    ], width=6),
            
                    dbc.Col([
                            html.H5('Values for each layer and category (for each language):', style={'text-align': 'center'}),
                            dcc.Dropdown(id='category'),
                            dcc.Dropdown(id='languages', multi=True), 
                            dcc.Graph(id='graph3'),
                    ], width=6),
                ], style={"padding-top": "3rem"})
                                
            ]),

            html.Div(children=[
                dbc.Row([
                    dbc.Col([
                            html.H5('Values for each layer and category (for certain language):', style={'text-align': 'center'}),
                            dcc.Dropdown(id='language_graph4'), 
                            dcc.Dropdown(id='categories_graph4', multi=True),
                            dcc.Graph(id='graph4'),
                    ], width=6),
                ], style={"padding-top": "3rem"})
                                
            ]),

            html.Div(children=[
                html.H3('Statistics on language families', style={'text-align': 'center'}),
                html.Br(),
                dcc.Dropdown(id='family_selection'),

                dbc.Row([
                    dbc.Col(id='card1', width=4),
                    dbc.Col(id='card2', width=4),
                    dbc.Col(id='card3', width=4),
                    ], style={"padding-top": "2rem", "padding-bottom": "2rem"}),
                    
                html.Div(children=[
                        dbc.Row([
                            dbc.Col([
                                    html.H5('Structure of the language family:', style={'text-align': 'center'}),
                                    dcc.Graph(id='treemap1'),
                                    ], width=5),
                            dbc.Col([
                                    html.H5('Average values by category:', style={'text-align': 'center'}),
                                    dcc.Graph(id='table1'),
                                    ], width=7),

                            ])  
                        ], ),
  
            ]),

            dbc.Row([
                    dbc.Col(dcc.Dropdown(id='quantity_of_languages'), width=8),
                    ], style={'margin-bottom': '3em'}),
            html.Div(id='graphs_for_family'),

            dcc.Graph(id='heatmap1'),
            dcc.Dropdown(id='category_heatmap2'),
            dcc.Graph(id='heatmap2'),

            ], justify="center")
                    
        ], style={"padding": "3rem"})

    app.run_server(debug=True, port=PORT)
