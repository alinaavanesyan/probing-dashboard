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
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html
from statistics import mean
import math
from collections import OrderedDict
import collections
import statistics

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

server = app.server

hits = glob("Probing_framework4/results/*/*/*.json", recursive=True)

lang_file = pd.read_csv('all_languages.csv', delimiter=';')

# extracting the information necessary for plots and storing it in dictionaries
all_categories = {}
middle_all_layers_family = {}
all_layers_lang = {}
count_occurrences = {}
lang_files = {}
model_names = []
for file_name in hits: 
    file = open(file_name)
    data_file = json.loads(file.read())
    lang = data_file['params']['task_language']
    model_name = data_file['params']['hf_model_name']
    if model_name not in model_names:
        model_names.append(model_name)
    a = lang_file.loc[lang_file['Codes'].isin([lang])]
    lang_full = a.iloc[0]['Language']
    family1 = a.iloc[0]['Family']
    if family1 == '-':
        family = '[Basque]*'
    else:
        family = family1
    if model_name not in lang_files.keys():
        lang_files[model_name] = {}
    if family not in lang_files[model_name].keys():
        lang_files[model_name][family] = {}
        lang_files[model_name][family][lang] = []
        lang_files[model_name][family][lang].append(file_name)
    else:
        if lang not in lang_files[model_name][family].keys():
            lang_files[model_name][family][lang] = []
            lang_files[model_name][family][lang].append(file_name)
        else:
            lang_files[model_name][family][lang].append(file_name)
    cat = data_file['params']['task_category']
    if model_name not in middle_all_layers_family.keys():
        middle_all_layers_family[model_name] = {}
    if family not in middle_all_layers_family[model_name].keys():
        middle_all_layers_family[model_name][family] = {}
        middle_all_layers_family[model_name][family]['f1'] = {}
        middle_all_layers_family[model_name][family]['accuracy'] = {}
    if model_name not in all_categories.keys():
        all_categories[model_name] = {}
    if family not in all_categories[model_name].keys():
        all_categories[model_name][family] = {}
        all_categories[model_name][family]['f1'] = {}
        all_categories[model_name][family]['accuracy'] = {}
    if cat not in all_categories[model_name][family]['f1'].keys():
        all_categories[model_name][family]['f1'][cat] = 0
        all_categories[model_name][family]['accuracy'][cat] = 0
    number = len(data_file['results']['test_score']['f1'])
    if model_name not in all_layers_lang.keys():
        all_layers_lang[model_name] = {}
    if lang_full not in all_layers_lang[model_name].keys():
        all_layers_lang[model_name][lang_full] = {}
        all_layers_lang[model_name][lang_full]['f1'] = {}
        all_layers_lang[model_name][lang_full]['accuracy'] = {}
    if cat not in all_layers_lang[model_name][lang_full]['f1'].keys():
        all_layers_lang[model_name][lang_full]['f1'][cat] = {}
        all_layers_lang[model_name][lang_full]['accuracy'][cat] = {}
    number_of_layers = len(data_file['results']['test_score']['f1'].keys())
    middle_value_for_language_f1 = 0
    middle_value_for_language_acc = 0
    for b in range(0,number_of_layers):
        middle_value_for_language_f1 += data_file['results']['test_score']['f1'][str(b)][0]
        middle_value_for_language_acc += data_file['results']['test_score']['accuracy'][str(b)][0]
        if b not in middle_all_layers_family[model_name][family]['f1'].keys():
            middle_all_layers_family[model_name][family]["f1"][b] = 0
            middle_all_layers_family[model_name][family]["accuracy"][b] = 0
        middle_all_layers_family[model_name][family]["f1"][b] += data_file['results']['test_score']['f1'][str(b)][0]
        middle_all_layers_family[model_name][family]["accuracy"][b] += data_file['results']['test_score']['accuracy'][str(b)][0]
        if b not in all_layers_lang[model_name][lang_full]['f1'][cat].keys():
            all_layers_lang[model_name][lang_full]['f1'][cat][b] = round(data_file['results']['test_score']['f1'][str(b)][0], 3)
            all_layers_lang[model_name][lang_full]['accuracy'][cat][b] = round(data_file['results']['test_score']['accuracy'][str(b)][0], 3)
    
    middle_value_for_language_f1 = round(middle_value_for_language_f1/number_of_layers, 3)
    middle_value_for_language_acc = round(middle_value_for_language_acc/number_of_layers, 3)
    
    all_categories[model_name][family]['f1'][cat] = round((all_categories[model_name][family]['f1'][cat] + middle_value_for_language_f1)/2, 3)
    all_categories[model_name][family]['accuracy'][cat] = round((all_categories[model_name][family]['accuracy'][cat] + middle_value_for_language_acc)/2, 3)
    
    if model_name not in count_occurrences.keys():
        count_occurrences[model_name] = {}
    if family not in count_occurrences[model_name].keys():
        count_occurrences[model_name][family] = 1
    else:
            count_occurrences[model_name][family] += 1



middle_values = {}
for model_name in all_categories.keys():
    middle_values[model_name] = {}
    for k in all_categories[model_name].keys():
        middle_values[model_name][k] = 0
        count = 0
        for key in all_categories[model_name][k]['f1']:
            count += 1
            middle_values[model_name][k] += all_categories[model_name][k]['f1'][key]
        middle_values[model_name][k] = round(middle_values[model_name][k]/count, 3)
        count = 0

size = {}
for model_name in all_categories.keys():
    families = all_categories[model_name].keys()
    size[model_name] = {}
    size[model_name]['number_of_languages']= {}
    size[model_name]['number_of_files']= {}
    for f in families:
        if f == '[Basque]*':
            family = '-'
        else:
            family = f
        number_of_languages = len(lang_file[(lang_file['Family'] == family)])
        size[model_name]['number_of_languages'][family] = number_of_languages
        number_of_files = 0
        if f == '[Basque]*':
            family = '[Basque]*'
        for language in lang_files[model_name][family].keys():
            number_of_files += len(lang_files[model_name][family][language])
        size[model_name]['number_of_files'][family] = number_of_files

structure = {}

for model_name in all_categories.keys():
    structure[model_name] = {}
    families = all_categories[model_name].keys()
    for f in families:
        lang = lang_file[(lang_file['Family'] == f)]
        lang_list = lang['Language'].tolist()
        structure[model_name][f] = lang_list

for model_name in middle_all_layers_family.keys():
    for family in middle_all_layers_family[model_name]:
        for el in middle_all_layers_family[model_name][family]["f1"].keys():
            middle_all_layers_family[model_name][family]["f1"][el] = round(middle_all_layers_family[model_name][family]["f1"][el]/count_occurrences[model_name][family], 3)
        for el in middle_all_layers_family[model_name][family]['accuracy'].keys():
            middle_all_layers_family[model_name][family]['accuracy'][el] = round(middle_all_layers_family[model_name][family]['accuracy'][el]/count_occurrences[model_name][family], 3)

middle_lang_and_cat = {}
for model_name in lang_files.keys():
    middle_lang_and_cat[model_name] = {}
    for family in lang_files[model_name].keys():
        for lang_code in lang_files[model_name][family]:
            for file_path in lang_files[model_name][family][lang_code]:
                file = open(file_path)
                data_file = json.loads(file.read())
                cat = data_file['params']['task_category']
                lang = data_file['params']['task_language']
                lang_full_name = lang_file.loc[lang_file['Codes'].isin([lang])]
                lang_full_name = lang_full_name.iloc[0]['Language']
                if family not in middle_lang_and_cat[model_name].keys():
                    middle_lang_and_cat[model_name][family] = {}
                if lang_full_name not in middle_lang_and_cat[model_name][family].keys():
                    middle_lang_and_cat[model_name][family][lang_full_name] = {'f1':{}, 'accuracy': {}}
                number_of_layers = len(data_file['results']['test_score']['f1'].keys())
                if cat not in middle_lang_and_cat[model_name][family][lang_full_name]['f1'].keys():
                    middle_lang_and_cat[model_name][family][lang_full_name]['f1'][cat] = 0
                    middle_lang_and_cat[model_name][family][lang_full_name]['accuracy'][cat]= 0
        
                    for b in range(0,number_of_layers):
                        middle_lang_and_cat[model_name][family][lang_full_name]['f1'][cat] += data_file["results"]["test_score"]["f1"][str(b)][0]
                        middle_lang_and_cat[model_name][family][lang_full_name]['accuracy'][cat] += data_file["results"]["test_score"]["accuracy"][str(b)][0]
                        
                    middle_lang_and_cat[model_name][family][lang_full_name]['f1'][cat] = round(middle_lang_and_cat[model_name][family][lang_full_name]['f1'][cat]/number_of_layers, 3)
                    middle_lang_and_cat[model_name][family][lang_full_name]['accuracy'][cat] = round(middle_lang_and_cat[model_name][family][lang_full_name]['accuracy'][cat]/number_of_layers, 3)
                else:
                    new_f1 = 0
                    new_accuracy = 0 
                    for b in range(0,number_of_layers):
                        new_f1 += data_file["results"]["test_score"]["f1"][str(b)][0]
                        new_accuracy += data_file["results"]["test_score"]["accuracy"][str(b)][0]
                    new_f1 = round(new_f1/number_of_layers, 3)
                    new_accuracy = round(new_accuracy/number_of_layers, 3)

                    middle_lang_and_cat[model_name][family][lang_full_name]['f1'][cat] = round((middle_lang_and_cat[model_name][family][lang_full_name]['f1'][cat] + new_f1)/2, 3)
                    middle_lang_and_cat[model_name][family][lang_full_name]['accuracy'][cat] = round((middle_lang_and_cat[model_name][family][lang_full_name]['accuracy'][cat] + new_accuracy)/2, 3)

cat_statistics = {}
for model_name in middle_lang_and_cat.keys():
    cat_statistics[model_name] = {}
    for family in middle_lang_and_cat[model_name].keys():
        cat_statistics[model_name][family] = {}
        for language in middle_lang_and_cat[model_name][family].keys():
            for category in middle_lang_and_cat[model_name][family][language]['f1']:
                if category not in cat_statistics[model_name][family]:
                    cat_statistics[model_name][family][category] = {}
                    cat_statistics[model_name][family][category][language] = middle_lang_and_cat[model_name][family][language]['f1'][category]
                else:
                    cat_statistics[model_name][family][category][language] = middle_lang_and_cat[model_name][family][language]['f1'][category]

cat_statistics_for_table = {}
for model_name in cat_statistics.keys():
    cat_statistics_for_table[model_name] = {}
    for family in cat_statistics[model_name].keys():
        cat_statistics_for_table[model_name][family] = {}
        for category in cat_statistics[model_name][family].keys():
            if len(list(cat_statistics[model_name][family][category].keys())) > 1:
                min_cat = min(cat_statistics[model_name][family][category].items(), key=lambda x: x[1])
                max_cat = max(cat_statistics[model_name][family][category].items(), key=lambda x: x[1])
                cat_statistics_for_table[model_name][family][category] = [min_cat, max_cat]
            else:
                language = list(cat_statistics[model_name][family][category].keys())[0]
                cat_statistics_for_table[model_name][family][category] = f'The category in the selected language family is represented by only one language/checked in one language - {language} - {cat_statistics[model_name][family][category][language]}'

all_layers_lang_middle = {}
for model_name in all_layers_lang.keys():
    all_layers_lang_middle[model_name] = {}
    for language in all_layers_lang[model_name].keys():
        all_layers_lang_middle[model_name][language] = {}
        for metric in all_layers_lang[model_name][language].keys():
            all_layers_lang_middle[model_name][language][metric] = {}
            number_of_categories = len(all_layers_lang[model_name][language][metric].keys())
            for category in all_layers_lang[model_name][language][metric].keys():
                number_of_layers = len(all_layers_lang[model_name][language][metric][category].keys())
                all_layers_lang_middle[model_name][language][metric] = {}
                for b in range(number_of_layers):
                    if b not in all_layers_lang_middle[model_name][language].keys():
                        all_layers_lang_middle[model_name][language][metric][b] = 0
                    all_layers_lang_middle[model_name][language][metric][b] += all_layers_lang[model_name][language][metric][category][b]
            
            for b in range(number_of_categories):
                all_layers_lang_middle[model_name][language][metric][b] = round(all_layers_lang_middle[model_name][language][metric][b]/number_of_categories, 3)


full_layers = {}
for file_name in hits: 
    file = open(file_name)
    data_file = json.loads(file.read())
    lang = data_file["params"]["task_language"]
    lang_full_name = lang_file.loc[lang_file['Codes'].isin([lang])]
    lang_full_name = lang_full_name.iloc[0]['Language']
    category = data_file['params']['task_category']
    model_name = data_file['params']['hf_model_name']
    number_of_layers = len(data_file['results']['test_score']['f1'].keys())
    if model_name not in full_layers.keys():
        full_layers[model_name] = {}
    metrics = data_file['results']['test_score'].keys()
    if lang_full_name not in full_layers[model_name].keys():
        full_layers[model_name][lang_full_name] = {}
        for metric in metrics:
            full_layers[model_name][lang_full_name][metric] = {}
    for metric in full_layers[model_name][lang_full_name].keys():
        if category not in full_layers[model_name][lang_full_name][metric].keys():
            full_layers[model_name][lang_full_name][metric][category] = {}
        for b in range(number_of_layers):
            full_layers[model_name][lang_full_name][metric][category][b] = round(data_file['results']['test_score'][str(metric)][str(b)][0], 3)

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

df_full_layers = pd.DataFrame(columns = ['Model', 'Language', 'Category', 'Metric', 'X', 'Y'])
for model_name in full_layers.keys():
    for language in full_layers[model_name].keys():
        for metric in full_layers[model_name][language].keys():
            cats = full_layers[model_name][language][metric].keys()
            for cat in cats:
                layers = full_layers[model_name][language][metric][cat]
                x = list(layers.keys())
                y = list(layers.values())
                models = [model_name] *len(x)
                languages = [language] *len(x)
                categories = [cat] *len(x)
                metrics = [metric] *len(x)
                df_temp = pd.DataFrame(columns = ['Model', 'Language', 'Category', 'Metric', 'X', 'Y'])
                df_temp['Model'] = models; df_temp['Language'] = languages; df_temp['Category'] = categories; df_temp['Metric'] = metrics
                df_temp['X'] = x; df_temp['Y'] = y
                frames = [df_full_layers, df_temp]
                df_full_layers = pd.concat(frames)

different_models = [{'label': str(i), 'value': str(i)} for i in model_names]

@app.callback(
    Output(component_id='graph1', component_property='figure'),
    [   Input(component_id='model_selection', component_property='value'),
        Input(component_id='parameter_graph1', component_property='value')
    ]
)
def update_output(model_name, parameter):
    df_graph1 = pd.DataFrame(list(middle_values[model_name].items()), columns = ['Family', 'Middle'])
    df_graph1['Size'] = size[model_name][parameter].values()
    fig1 = px.scatter(df_graph1,
    x='Family', 
    y='Middle', 
    size='Size'
    )
    fig1.update_layout(hovermode='x unified')
    return fig1

@app.callback(
    Output(component_id='note_graph1', component_property='children'),
    Input(component_id='model_selection', component_property='value'),
)
def update_output(model_name):
    res = str()
    if '[Basque]*' in middle_values[model_name].keys():
       res = '*Баскский язык - язык изолят, он не входит ни в одну из языковых групп'
    return res

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
    fig2 = go.Figure()
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
            fig2 = px.line(df_graph2, x = 'X', y = 'Y', color='Language', labels={'X':'Layer number', 'Y': 'Value'})
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
        fig2 = px.line(df_graph2, x = 'X', y = 'Y', color='Language', labels={'X':'Layer number', 'Y': 'Value'})
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
    df_graph3 = pd.DataFrame(columns = ['Language', 'x', 'y'])
    fig3 = go.Figure()
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
        fig3 = px.line(df_graph3, x = 'X', y = 'Y', color='Language',labels={'X':'Layer number', 'Y': 'Value'})
    else:
        df_temp = df_full_layers[df_full_layers['Model'].isin([model_name])]
        df_temp = df_temp[df_temp['Language'].isin([languages])]     
        df_temp = df_temp[df_temp['Metric'].isin(['f1'])]
        df_temp = df_temp[df_temp['Category'].isin([category])]
        df_temp.pop('Model'); df_temp.pop('Metric')   
        frames = [df_temp, df_graph3]
        df_graph3 = pd.concat(frames)
        fig3 = px.line(df_graph3, x = 'X', y = 'Y', color='Language', labels={'X':'Layer number', 'Y': 'Value'})
    return fig3

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
    df_info_for_graphs = pd.DataFrame(columns = ['Language', 'Category', 'X', 'Y'])
    df_similar_graph1 = pd.DataFrame(columns = ['Language', 'Category', 'X', 'Y'])
    df_dissimilar_graph2 = pd.DataFrame(columns = ['Language', 'Category', 'X', 'Y'])
    df_incomparable_graph3 = pd.DataFrame(columns = ['Language', 'Category', 'X', 'Y'])
    family_structure = structure[model_name][family]
    df_spec = df_full_layers[df_full_layers['Model'].isin([model_name])]
    df_spec = df_spec[df_spec['Metric'].isin(['f1'])]
    df_spec.pop('Model'); df_spec.pop('Metric')
    for language in family_structure:
        df_temp = df_spec[df_spec['Language'].isin([language])]
        frames = [df_info_for_graphs, df_temp]
        df_info_for_graphs = pd.concat(frames)
    distances = {}
    for category in df_info_for_graphs['Category'].unique().tolist():
        a = df_info_for_graphs[df_info_for_graphs['Category'].isin([category])]
        if len(a['Language'].unique().tolist()) > 3:
            curves = {}
            for index, raw in df_info_for_graphs.iterrows():
                if raw['Category'] not in curves.keys():
                    curves[raw['Category']] = {}
                if raw['Language'] not in curves[raw['Category']].keys():
                    curves[raw['Category']][raw['Language']] = []
                point = (raw['X'], raw['Y'])
                curves[raw['Category']][raw['Language']].append(point)
            distances[category] = {}
            pattern_line = []
            nam_line = list(curves[category].keys())
            number_of_layers = len(df_info_for_graphs['X'].unique().tolist())
            for b in range(number_of_layers):
                for_pattern = []
                for language in curves[category].keys():
                    for_pattern.append(curves[category][language][b][1])
                for_median = sorted(for_pattern)
                median = statistics.median(for_median)
                pattern_line.append((b, round(median, 3)))
            for name in nam_line:
                distances[category][name] = frechet_distance(curves[category][name], pattern_line)
            distances1 = dict(sorted(distances[category].items(), key=lambda x: -x[1], reverse=True))
            distances2 = dict(sorted(distances[category].items(), key=lambda x: -x[1]))
            distances1_names = list(distances1.keys())
            distances2_names = list(distances2.keys())            
            if quantity:
                if quantity > 1:
                    for b in range(quantity):
                        if b >= len(distances1_names):
                            b = len(distances1_names) - 1

                        df_temp = df_info_for_graphs[df_info_for_graphs['Language'].isin([distances1_names[b]])]
                        df_temp = df_temp.loc[df_temp['Category'].isin([category])]
                        frames = [df_similar_graph1, df_temp]
                        df_similar_graph1 = pd.concat(frames)

                        df_temp = df_info_for_graphs[df_info_for_graphs['Language'].isin([distances2_names[b]])]
                        df_temp = df_temp.loc[df_temp['Category'].isin([category])]
                        frames = [df_dissimilar_graph2, df_temp]
                        df_dissimilar_graph2 = pd.concat(frames)
                else:
                    b = 0
                    df_temp = df_info_for_graphs[df_info_for_graphs['Language'].isin([distances1_names[b]])]
                    df_temp = df_temp.loc[df_temp['Category'].isin([category])]
                    frames = [df_similar_graph1, df_temp]
                    df_similar_graph1 = pd.concat(frames)

                    df_temp = df_info_for_graphs[df_info_for_graphs['Language'].isin([distances2_names[b]])]
                    df_temp = df_temp.loc[df_temp['Category'].isin([category])]
                    frames = [df_dissimilar_graph2, df_temp]
                    df_dissimilar_graph2 = pd.concat(frames)
                        
        else:
            df_temp = df_info_for_graphs[df_info_for_graphs['Category'].isin([category])]
            frames = [df_incomparable_graph3, df_temp]
            df_incomparable_graph3 = pd.concat(frames)

    names_for_legend = df_similar_graph1['Language'].unique().tolist() + df_dissimilar_graph2['Language'].unique().tolist() + df_incomparable_graph3['Language'].unique().tolist()
    names_for_legend = sorted(list(set(names_for_legend)))
    colors_for_legend = {}
    for i in range(len(names_for_legend)):
        try:
            colors_for_legend[names_for_legend[i]] = px.colors.qualitative.Dark24[i]
        except:
            count = 0
            color = str()
            while True:
                if color == '':
                    if px.colors.qualitative.Light24[count] not in colors_for_legend.values():
                        color = px.colors.qualitative.Light24[count]
                        colors_for_legend[names_for_legend[i]] = color
                    else:
                        count += 1
                else:
                    break
    graphs_for_div = []
    similar_graphs_for_col = []
    dissimilar_graphs_for_col = []
    incomparable_graphs_for_col = []
    if df_similar_graph1.empty == False:
        for category in df_similar_graph1['Category'].unique().tolist():
            df_temp = df_similar_graph1[df_similar_graph1['Category'].isin([category])]
            fig1 = px.line(df_temp, x='X', y='Y', color='Language', 
                        labels={'X':'Layer number', 'Y': 'Value'}, title=str(category))
            fig1['layout'].update(height=350)
            row = dcc.Graph(figure=fig1)
            if len(similar_graphs_for_col) == 0:
                similar_graphs_for_col.append(html.H5('The most similar trends by category:',  style={'font-weight': 'bold', 'text-align': 'center'}))
            similar_graphs_for_col.append(row)
    if df_dissimilar_graph2.empty == False:
        for category in df_dissimilar_graph2['Category'].unique().tolist():
            df_temp = df_dissimilar_graph2[df_dissimilar_graph2['Category'].isin([category])]
            fig2 = px.line(df_temp, x='X', y='Y', color='Language', 
                        labels={'X':'Layer number', 'Y': 'Value'}, title=str(category))
            fig2['layout'].update(height=350)
            row = dcc.Graph(figure=fig2)
            if len(dissimilar_graphs_for_col) == 0:
                dissimilar_graphs_for_col.append(html.H5('The most dissimilar trends by category:', style={'font-weight': 'bold', 'text-align': 'center'}))
            dissimilar_graphs_for_col.append(row)
    if df_incomparable_graph3.empty == False:
        for category in df_incomparable_graph3['Category'].unique().tolist():
            df_temp = df_incomparable_graph3[df_incomparable_graph3['Category'].isin([category])]
            fig3 = px.line(df_temp, x='X', y='Y', color='Language', 
                        labels={'X':'Layer number', 'Y': 'Value'}, title=str(category))
            fig3['layout'].update(height=350)
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

if __name__ == "__main__":  
    app.layout = html.Div(children=[
    dbc.Row([

            html.Div(children=[
                dbc.Row([
                    dbc.Col([
                        html.H1('Probing results'),
                    ], width=6),

                    dbc.Col([
                        html.H5('Choose a model:'),
                    ], width=3),

                    dbc.Col([
                        dcc.Dropdown(
                            id='model_selection',
                            options = different_models,
                            value = model_names[0], 
                            style={'width': '100%' }),
                    ], width=3),
                ]),
            ]),
                
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                    options=[
                    {'label': 'number_of_languages', 'value': 'number_of_languages'},
                    {'label': 'number_of_files', 'value': 'number_of_files'}],
                    value = 'number_of_languages',
                    id='parameter_graph1', style={'width': '60%'})
                ], width=3)
            ]),
            dcc.Graph(id='graph1'),
            html.P(id='note_graph1', style={"padding-bottom": "2rem"}),
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
                ])
                                
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
            html.Div(id='graphs_for_family')

            ], justify="center")
                    
        ], style={"padding": "3rem"})

    app.run_server(debug=True)
