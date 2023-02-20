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

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

server = app.server

hits = glob("Probing_framework4/results/*/*/*.json", recursive=True)

lang_file = pd.read_csv('all_languages.csv', delimiter=';')

all_categories = {}
count2 = {}
all_layers = {}
all_layers_lang = {}
lang_files = {}

for file_name in hits: 
    file = open(file_name)
    data_file = json.loads(file.read())
    lang = data_file["params"]["task_language"]
    a = lang_file.loc[lang_file['Codes'].isin([lang])]
    lang_full = a.iloc[0]['Language']
    a = a.iloc[0]['Family']
    if a not in lang_files.keys():
        lang_files[a] = {}
        lang_files[a][lang] = []
        lang_files[a][lang].append(file_name)
    else:
        if lang not in lang_files[a].keys():
            lang_files[a][lang] = []
            lang_files[a][lang].append(file_name)
        else:

            lang_files[a][lang].append(file_name)

    cat = data_file["params"]["task_category"]
    if a not in all_layers.keys():
        all_layers[a] = {}
        all_layers[a]['f1'] = {}
        all_layers[a]['accuracy'] = {}
    if a not in all_categories.keys():
        all_categories[a] = {}
        all_categories[a]['f1'] = {}
        all_categories[a]['accuracy'] = {}
    if cat not in all_categories[a]['f1'].keys():
        all_categories[a]['f1'][cat] = 0
        all_categories[a]['accuracy'][cat] = 0
    number = len(data_file["results"]["test_score"]["f1"])
    if lang_full not in all_layers_lang.keys():
        all_layers_lang[lang_full] = {}
        all_layers_lang[lang_full]['f1'] = {}
        all_layers_lang[lang_full]['accuracy'] = {}
    if cat not in all_layers_lang[lang_full]['f1'].keys():
        all_layers_lang[lang_full]['f1'][cat] = {}
        all_layers_lang[lang_full]['accuracy'][cat] = {}
    for b in range(0,24):
        all_categories[a]['f1'][cat] += data_file["results"]["test_score"]["f1"][str(b)][0]
        all_categories[a]['accuracy'][cat] += data_file["results"]["test_score"]["accuracy"][str(b)][0]
        if b not in all_layers[a]['f1'].keys():
            all_layers[a]["f1"][b] = 0
            all_layers[a]["accuracy"][b] = 0
        all_layers[a]["f1"][b] += data_file["results"]["test_score"]["f1"][str(b)][0]
        all_layers[a]["accuracy"][b] += data_file["results"]["test_score"]["accuracy"][str(b)][0]
        if b not in all_layers_lang[lang_full]['f1'][cat].keys():
            all_layers_lang[lang_full]['f1'][cat][b] = round(data_file["results"]["test_score"]["f1"][str(b)][0], 3)
            all_layers_lang[lang_full]['accuracy'][cat][b] = round(data_file["results"]["test_score"]["accuracy"][str(b)][0], 3)
    all_categories[a]['f1'][cat] = round(all_categories[a]['f1'][cat]/24, 3)
    all_categories[a]['accuracy'][cat] = round(all_categories[a]['accuracy'][cat]/24, 3)
    if a not in count2.keys():
        count2[a] = 1
    else:
            count2[a] += 1



middle_values = {}
for k in all_categories.keys():
    middle_values[k] = 0
    count = 0
    for key in all_categories[k]['f1']:
        count += 1
        middle_values[k] += all_categories[k]['f1'][key]
    middle_values[k] = round(middle_values[k]/count, 3)
    count = 0

middle_values['[Basque]*'] = middle_values.pop('-')

size = {}
families = all_categories.keys()
for f in families:
    number = len(lang_file[(lang_file['Family'] == f)])
    size[f] = number

df = pd.DataFrame(list(middle_values.items()), columns = ['Family', 'Middle'])
df['Size'] = size.values()

structure = {}

families = all_categories.keys()
for f in families:
    lang = lang_file[(lang_file['Family'] == f)]
    lang_list = lang['Language'].tolist()
    structure[f] = lang_list


for k in all_layers.keys():
    for el in all_layers[k]["f1"].keys():
        all_layers[k]["f1"][el] = round(all_layers[k]["f1"][el]/count2[k], 3)
    for el in all_layers[k]["accuracy"].keys():
        all_layers[k]["accuracy"][el] = round(all_layers[k]["accuracy"][el]/count2[k], 3)

lang_and_cat = {}
for k in lang_files.keys():
    for language in lang_files[k]:
        for file_path in lang_files[k][language]:
            file = open(file_path)
            data_file = json.loads(file.read())
            cat = data_file["params"]["task_category"]
            lang = data_file["params"]["task_language"]
            a = lang_file.loc[lang_file['Codes'].isin([lang])]
            a = a.iloc[0]['Language']
            if k not in lang_and_cat.keys():
                lang_and_cat[k] = {}
            if a not in lang_and_cat[k].keys():
                lang_and_cat[k][a] = {'f1':{}, 'accuracy': {}}
            if cat not in lang_and_cat[k][a]['f1'].keys():
                lang_and_cat[k][a]['f1'][cat] = 0
                lang_and_cat[k][a]['accuracy'][cat]= 0
                for b in range(0,24):
                    lang_and_cat[k][a]['f1'][cat] += data_file["results"]["test_score"]["f1"][str(b)][0]
                    lang_and_cat[k][a]['accuracy'][cat] += data_file["results"]["test_score"]["accuracy"][str(b)][0]
                    
                lang_and_cat[k][a]['f1'][cat] = round(lang_and_cat[k][a]['f1'][cat]/24, 3)
                lang_and_cat[k][a]['accuracy'][cat] = round(lang_and_cat[k][a]['accuracy'][cat]/24, 3)
            else:
                new_f1 = 0
                new_accuracy = 0 
                for b in range(0,24):
                    new_f1 += data_file["results"]["test_score"]["f1"][str(b)][0]
                    new_accuracy += data_file["results"]["test_score"]["accuracy"][str(b)][0]
                new_f1 = round(new_f1/24, 3)
                new_accuracy = round(new_accuracy/24, 3)

                lang_and_cat[k][a]['f1'][cat] = round((lang_and_cat[k][a]['f1'][cat] + new_f1)/2, 3)
                lang_and_cat[k][a]['accuracy'][cat] = round((lang_and_cat[k][a]['accuracy'][cat] + new_accuracy)/2, 3)

cat_statistics = {}
for k in lang_and_cat.keys():
    cat_statistics[k] = {}
    for l in lang_and_cat[k].keys():
        for category in lang_and_cat[k][l]['f1']:
            if category not in cat_statistics[k]:
                cat_statistics[k][category] = {}
                cat_statistics[k][category][l] = lang_and_cat[k][l]['f1'][category]
            else:
                cat_statistics[k][category][l] = lang_and_cat[k][l]['f1'][category]

cat_itog = {}
for fam in cat_statistics.keys():
    cat_itog[fam] = {}
    for category in cat_statistics[fam].keys():
        if len(list(cat_statistics[fam][category].keys())) > 1:
            min_cat = min(cat_statistics[fam][category].items(), key=lambda x: x[1])
            max_cat = max(cat_statistics[fam][category].items(), key=lambda x: x[1])
            cat_itog[fam][category] = [min_cat, max_cat]
        else:
            lang = list(cat_statistics[fam][category].keys())[0]
            cat_itog[fam][category] = f'Категория в данной языковой семье представлена (или проверена) только в одном языке - {lang} - {cat_statistics[fam][category][lang]}'


df2 = pd.DataFrame(columns = ['Языковая семья', 'Категория', 'Минимальное значение', 'Максимальное значение'])
for fam in cat_itog.keys():
    for category in cat_itog[fam].keys():
        if isinstance(cat_itog[fam][category], list):
            row = [fam, category, cat_itog[fam][category][0], cat_itog[fam][category][1]]
            df2.loc[len(df2.index)] = row
        else:
            row = [fam, category, cat_itog[fam][category], cat_itog[fam][category]]
            df2.loc[len(df2.index)] = row




all_layers_lang_middle = {}
for k in all_layers_lang.keys():
    all_layers_lang_middle[k] = {}
    for b in range(24):
        all_layers_lang_middle[k][b] = 0
        for n in all_layers_lang[k]['f1'].keys():
            all_layers_lang_middle[k][b] += all_layers_lang[k]['f1'][n][b]
            all_layers_lang_middle[k][b] = all_layers_lang_middle[k][b]/2
        all_layers_lang_middle[k][b] = round(all_layers_lang_middle[k][b], 3)


full_layers = {}
for file_name in hits: 
    file = open(file_name)
    data_file = json.loads(file.read())
    lang = data_file["params"]["task_language"]
    a = lang_file.loc[lang_file['Codes'].isin([lang])]
    a = a.iloc[0]['Language']
    # a = a.iloc[0]['Family']
    cat = data_file["params"]["task_category"]
    if a not in full_layers.keys():
        full_layers[a] = {}
        full_layers[a]['f1'] = {}
        full_layers[a]['accuracy'] = {}
    if cat not in full_layers[a]['f1'].keys():
        full_layers[a]['f1'][cat] = {}
        full_layers[a]['accuracy'][cat] = {}

    for b in range(24):
        full_layers[a]['f1'][cat][b] = round(data_file["results"]["test_score"]["f1"][str(b)][0], 3)
        full_layers[a]['accuracy'][cat][b] = round(data_file["results"]["test_score"]["accuracy"][str(b)][0], 3)


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
 

df1 = pd.DataFrame(columns = ['Name', 'x', 'y', 'Category'])
for element in full_layers.keys():
    cats = full_layers[element]['f1'].keys()
    for cat in cats:
        name = []
        cats = []
        layers = full_layers[str(element)]['f1'][cat]
        graph_data = {'Layers': list(layers.keys()), 
            'Middle': list(layers.values())}
        graph_data = pd.DataFrame(graph_data)
        x = graph_data['Layers']
        y = graph_data['Middle']
        for i in x:
            name.append(element)
            cats.append(cat)
        df_temp = pd.DataFrame(columns = ['Name', 'x', 'y', 'Category'])
        df_temp['Name'] = name
        df_temp['x'] = x
        df_temp['y'] = y
        df_temp['Category'] = cats
        frames = [df1, df_temp]
        df1 = pd.concat(frames)

fig1 = px.scatter(df,
    x='Family', 
    y='Middle', 
    size='Size'
)


def graph1(k):
    rows = df2.loc[df2['Языковая семья'].isin([k])]
    a = rows['Категория']
    b = rows['Минимальное значение']
    c = rows['Максимальное значение']

    fig1 = go.Figure(data=[go.Table(header=dict(values=['Категория', 'Минимальное значение', 'Максимальное значение']),
                    cells=dict(values=[a, b, c]))
                        ])
    fig1.update_layout(margin=dict(l=20, r=20, t=20, b=20),)

    return fig1
    
def graph2(k):
    family = k
    labels = []
    labels.append(family)
    parents = []
    parents.append('')
    values = []
    values.append(0)


    for k in structure[family]:
        try:
            labels.append(k)
            parents.append(family)

            code = lang_file.loc[lang_file['Language'].isin([k])]
            code = code.iloc[0]['Codes']
            
            count = len(lang_files[family][code])
            values.append(count)
        except:
            value = f'Язык {k} входит в языковую семью, но не представлен в файлах с результатами пробинга'

    values[0] = sum(values)

    fig2 =go.Figure(go.Treemap(
        labels = labels,
        parents= parents,
        values= values,
        root = None,
    ))

    fig2.update_layout(
        font_size=20,
        margin = dict(t=20, l=15, r=20, b=20)
    )

    return fig2

@app.callback(
    Output('graph1', 'figure'),
    Input('demo-dropdown', 'value')
)

def update_output(value):
    return graph1(value)

@app.callback(
    Output('graph2', 'figure'),
    Input('demo-dropdown', 'value')
)

def update_output(value):
    return graph2(value)

@app.callback(
    Output('text1', 'children'),
    Input('demo-dropdown', 'value')
)

def update_output(value):
    min_count = 0
    min_count = min(all_categories[value]['f1'].items(), key=lambda x: x[1])
    
    card1 = dbc.Card(
        [
                       
            dbc.CardBody(
            [
                html.H2(f'{str(min_count[0])}'),
                html.P("Самая плохо распознаваемая моделью категория", className="card-text"),
            
            ]

            ),
        ],
        body=True,
        color="primary",
        inverse=True,
    )
    
    return card1

@app.callback(
    Output('text2', 'children'),
    Input('demo-dropdown', 'value')
)

def update_output(value):
    max_count = 0
    max_count = max(all_categories[value]['f1'].items(), key=lambda x: x[1])
    
    card2 = dbc.Card(
        [
                       
            dbc.CardBody(
            [
                html.H2(f'{str(max_count[0])}'),
                html.P("Самая хорошо распознаваемая моделью категория", className="card-text"),
            
            ]

            ),
        ],
        body=True,
        color="primary",
        inverse=True,
    )


    return card2


@app.callback(
    Output('text3', 'children'),
    Input('demo-dropdown', 'value')
)

def update_output(value):

    card3 = dbc.Card(
        [
                       
            dbc.CardBody(
            [
                html.H2(f'{len(structure[value])}'),
                html.P("Языков в данной языковой семье", className="card-text"),
            
            ]

            ),
        ],
        body=True,
        color="primary",
        inverse=True,
    )
    return card3

@app.callback(
    Output('graph', 'figure'),
    [Input(component_id='dropdown', component_property='value')],

)


def update_output(value):
    df1 = pd.DataFrame(columns = ['Name', 'x', 'y'])
    if isinstance(value, list):
        for element in value:
            name = []
            layers = all_layers[str(element)]['f1']
            graph_data = {'Layers': list(layers.keys()), 
                'Middle': list(layers.values())}
            graph_data = pd.DataFrame(graph_data)
            x = graph_data['Layers']
            y = graph_data['Middle']
            for i in x:
                name.append(element)
            df = pd.DataFrame(columns = ['Name', 'x', 'y'])
            df['Name'] = name
            df['x'] = x
            df['y'] = y
            frames = [df1, df]
            df1 = pd.concat(frames)
            fig = px.line(df1, x = 'x', y = 'y', color='Name')
    elif value == ['']:
        fig = go.Figure()
    else:
        name = []
        layers = all_layers[value]['f1']
        graph_data = {'Layers': list(layers.keys()), 
            'Middle': list(layers.values())}
        graph_data = pd.DataFrame(graph_data)
        x = graph_data['Layers']
        y = graph_data['Middle']
        for i in x:
            name.append(value)
        df = pd.DataFrame(columns = ['Name', 'x', 'y'])
        df['Name'] = name
        df['x'] = x
        df['y'] = y
        frames = [df1, df]
        df1 = pd.concat(frames)
        fig = px.line(df1, x = 'x', y = 'y', color='Name')
    return fig


@app.callback(
    Output('graph3', 'figure'),
    [Input(component_id='demo-dropdown', component_property='value')],

)

def update_output(family):
    df_spec = pd.DataFrame(columns = ['Name', 'x', 'y', 'Category'])
    df_spec2 = pd.DataFrame(columns = ['Name', 'x', 'y', 'Category'])
    new_lst = structure[family]

    for el in new_lst:
        df_temp = df1.loc[df1['Name'].isin([el])]
        frames = [df_spec, df_temp]
        df_spec = pd.concat(frames)

    for cat in df_spec['Category'].unique().tolist():
        a = df_spec.loc[df_spec['Category'].isin([cat])]
        
        if len(a['Name'].unique().tolist()) > 2:
            curves = {}

            for index, raw in df_spec.iterrows():
                if raw['Category'] not in curves.keys():
                    curves[raw['Category']] = {}
                if raw['Name'] not in curves[raw['Category']].keys():
                    curves[raw['Category']][raw['Name']] = []
                point = (raw['x'], raw['y'])
                curves[raw['Category']][raw['Name']].append(point)

            distances = {}
            distances[cat] = {}
            nam_line = list(curves[cat].keys())
            for b in range(len(curves[cat].keys())):
                nam_line_spec = list(curves[cat].keys())
                del nam_line_spec[b]
                line_1 = curves[cat][nam_line[b]]
                lang_1 = nam_line[b]
                for name in nam_line_spec:
                    line_2 = curves[cat][name]
                    lang_2 = name
                    compar_name1 = f'{lang_1}:{lang_2}'
                    compar_name2 = f'{lang_2}:{lang_1}'
                    if compar_name1 and compar_name2 not in distances[cat].keys():
                        distances[cat][compar_name1] = frechet_distance(line_1, line_2)

            distances = dict(sorted(distances[cat].items(), key=lambda x: -x[1]))
            compar_distances = {}
            count = 0
            dist_len = 0.8*len(distances.keys())
            pattern1 = int()
            for k in distances.keys():
                count += 1
                if pattern1 == 0:
                    pattern1 =  1.2*distances[k]
                n_index = k.index(':')
                lang1 = k[0:n_index]
                lang2 = k[n_index+1:]
                if distances[k] <= pattern1:
                    if count <= dist_len:
                        if lang1 not in compar_distances.keys():
                            compar_distances[lang1] = []
                            compar_distances[lang1].append(lang1)
                            compar_distances[lang1].append(lang2)
                        else:
                            compar_distances[lang1].append(lang2)
                        if lang2 not in compar_distances.keys():
                            compar_distances[lang2] = []
                            compar_distances[lang2].append(lang2)
                            compar_distances[lang2].append(lang1)
                        else:
                            compar_distances[lang2].append(lang1)
            compar_distances = dict(sorted(compar_distances.items(), key=lambda item: len(item[1])))
            [last] = collections.deque(compar_distances, maxlen=1)
            need_lang = compar_distances[last]
            
            if len(need_lang) >= 6:
                need_lang = need_lang[:6]

            for el in need_lang:
                b = df_spec2.loc[df_spec2['Category'].isin([cat])]
                b = b['Name'].unique().tolist()
                if el not in b:
                    df_temp = df_spec.loc[df_spec['Name'].isin([el])]
                    df_temp = df_temp.loc[df_temp['Category'].isin([cat])]
                    frames = [df_spec2, df_temp]
                    df_spec2 = pd.concat(frames)
             
        else:
            df_temp = df_spec[df_spec['Category'].isin([cat])]
            frames = [df_spec2, df_temp]
            df_spec2 = pd.concat(frames)


    fig = px.line(df_spec2, x='x', y='y', color='Name', facet_row='Category', facet_row_spacing=0.022)
    fig['layout'].update(height=2500)
    for annotation in fig['layout']['annotations']: 
        annotation['textangle']= 0
        annotation['x']-= 0.6
        annotation['y'] += 0.024
        annotation['font']=dict(size = 14)
    return fig


@app.callback(
    Output('graph4', 'figure'),
    [Input(component_id='demo-dropdown', component_property='value')],

)

def update_output(family):
    df_spec = pd.DataFrame(columns = ['Name', 'x', 'y', 'Category'])
    df_spec3 = pd.DataFrame(columns = ['Name', 'x', 'y', 'Category'])
    new_lst = structure[family]

    for el in new_lst:
        df_temp = df1.loc[df1['Name'].isin([el])]
        frames = [df_spec, df_temp]
        df_spec = pd.concat(frames)

    for cat in df_spec['Category'].unique().tolist():
        a = df_spec.loc[df_spec['Category'].isin([cat])]
        
        if len(a['Name'].unique().tolist()) > 2:
            curves = {}

            for index, raw in df_spec.iterrows():
                if raw['Category'] not in curves.keys():
                    curves[raw['Category']] = {}
                if raw['Name'] not in curves[raw['Category']].keys():
                    curves[raw['Category']][raw['Name']] = []
                point = (raw['x'], raw['y'])
                curves[raw['Category']][raw['Name']].append(point)

            distances = {}
            distances[cat] = {}
            nam_line = list(curves[cat].keys())
            for b in range(len(curves[cat].keys())):
                nam_line_spec = list(curves[cat].keys())
                del nam_line_spec[b]
                line_1 = curves[cat][nam_line[b]]
                lang_1 = nam_line[b]
                for name in nam_line_spec:
                    line_2 = curves[cat][name]
                    lang_2 = name
                    compar_name1 = f'{lang_1}:{lang_2}'
                    compar_name2 = f'{lang_2}:{lang_1}'
                    if compar_name1 and compar_name2 not in distances[cat].keys():
                        distances[cat][compar_name1] = frechet_distance(line_1, line_2)

            distances = dict(sorted(distances[cat].items(), key=lambda x: -x[1], reverse=True))
            compar_distances2 = {}
            dist_len = 0.8*len(distances.keys())
            pattern2 = int()
            count = 0
            for k in distances.keys():
                count += 1
        
                if pattern2 == 0:
                    pattern2 =  0.8*distances[k]
                n = k.index(':')
                lang1 = k[0:n]
                lang2 = k[n+1:]

                if distances[k] >= pattern2:
                    if count <= dist_len:
                        if lang1 not in compar_distances2.keys():
                            compar_distances2[lang1] = []
                            compar_distances2[lang1].append(lang1)
                            compar_distances2[lang1].append(lang2)
                        else:
                            if lang2 not in compar_distances2[lang1]:
                                compar_distances2[lang1].append(lang2)
                        if lang2 not in compar_distances2.keys():
                            compar_distances2[lang2] = []
                            compar_distances2[lang2].append(lang2)
                            compar_distances2[lang2].append(lang1)
                        else:
                            if lang1 not in compar_distances2[lang2]:
                                compar_distances2[lang2].append(lang1)
            
            compar_distances2 = dict(sorted(compar_distances2.items(), key=lambda item: len(item[1])))
            [last] = collections.deque(compar_distances2, maxlen=1)
            need_lang = compar_distances2[last]
            if len(need_lang) >= 6:
                need_lang = need_lang[:6]

            for el in need_lang:
                b = df_spec3.loc[df_spec3['Category'].isin([cat])]
                b = b['Name'].unique().tolist()
                if el not in b:
                    df_temp = df_spec.loc[df_spec['Name'].isin([el])]
                    df_temp = df_temp.loc[df_temp['Category'].isin([cat])]
                    frames = [df_spec3, df_temp]
                    df_spec3 = pd.concat(frames)
                
        else:
            df_temp = df_spec[df_spec['Category'].isin([cat])]
            frames = [df_spec3, df_temp]
            df_spec3 = pd.concat(frames)

    fig = px.line(df_spec3, x='x', y='y', color='Name', facet_row='Category', facet_row_spacing=0.022)
    fig['layout'].update(height=2500)
    for annotation in fig['layout']['annotations']: 
        annotation['textangle']= 0
        annotation['x']-= 0.6
        annotation['y'] += 0.024
        annotation['font']=dict(size = 14)

    return fig

all_cats = []
for fam in all_categories.keys():
    for cat in all_categories[fam]['f1'].keys():
        if cat not in all_cats:
            all_cats.append(cat)

@app.callback(
    [
    Output(component_id='languages', component_property='options'),
    Output(component_id='languages', component_property='value'),
    ],

    [Input(component_id='category', component_property='value')],

)

def update_output(cat):
    df_temp = df1[df1['Category'].isin([cat])]
    list_of_lang = df_temp['Name'].unique().tolist()
    return [{'label': str(i), 'value': str(i)} for i in list_of_lang], list_of_lang[0]


@app.callback(
        Output('graph7', 'figure'),
    [
        Input(component_id='category', component_property='value'),
        Input(component_id='languages', component_property='value')
    ],

)

def every_language(category, languages):
    df = pd.DataFrame(columns = ['Name', 'x', 'y'])
    if isinstance(languages, list):
        count = -1
        for element in languages:
            count += 1
            df_temp = df1[df1['Name'].isin([element])]
            df_temp = df_temp[df_temp['Category'].isin([category])]
            frames = [df_temp, df]
            df = pd.concat(frames)
        fig = px.line(df, x = 'x', y = 'y', color='Name')
    else:
        df_temp = df1[df1['Name'].isin([languages])]
        df_temp = df_temp[df_temp['Category'].isin([category])]
        frames = [df_temp, df]
        df = pd.concat(frames)
        fig = px.line(df, x = 'x', y = 'y', color='Name')
    return fig
    

if __name__ == "__main__":  
    app.layout = html.Div(children=[
    dbc.Row([

            dbc.Col([

                    html.H1("Probing results"),
                    html.Hr(),
                    dcc.Graph(figure=fig1),
                    html.P("*Баскский язык - язык изолят, он не входит ни в одну из языковых групп", style={"padding-bottom": "2rem"}),
            ]),

            html.Div(children=[
                dbc.Row([
                    dbc.Col([
                        html.H5('Средние значения по всем категориям (статистика по языковым семьям):', style={'text-align': 'center'}),
                        dcc.Dropdown(
                            id='dropdown',

                            options = [
                                {'label': 'Afro-Asiatic', 'value': 'Afro-Asiatic'},
                                {'label': 'Arawakan', 'value': 'Arawakan'},
                                {'label': 'Atlantic-Congo', 'value': 'Atlantic-Congo'},
                                {'label': 'Austronesian', 'value': 'Austronesian'},
                                {'label': 'Dravidian', 'value': 'Dravidian'},
                                {'label': 'Eskimo-Aleut', 'value': 'Eskimo-Aleut'},
                                {'label': 'Indo-European', 'value': 'Indo-European'},
                                {'label': 'Mande', 'value': 'Mande'},
                                {'label': 'Pama-Nyungan', 'value': 'Pama-Nyungan'},
                                {'label': 'Sino-Tibetan', 'value': 'Sino-Tibetan'},
                                {'label': 'Tungusic', 'value': 'Tungusic'},
                                {'label': 'Tupian', 'value': 'Tupian'},
                                {'label': 'Turkic', 'value': 'Turkic'},
                                {'label': 'Uralic', 'value': 'Uralic'},
                            ],

                            value = 'Afro-Asiatic',    
                        
                            multi=True
                        ),
                        html.Div(dcc.Graph(id='graph')),
                    ]),
            
                    dbc.Col([
                            html.H5('Значения по каждому слою и категории (по языкам):', style={'text-align': 'center'}),
                            dcc.Dropdown(id='category', options = [{'label': str(i), 'value': str(i)} for i in all_cats], value = all_cats[0]), 
                            
                            dcc.Dropdown(id='languages', multi=True), 

                            dcc.Graph(id='graph7'),
                            ]),

                    ])
                                
            ]),

            html.Div(children=[
                html.H3('Статистика по языковым семьям', style={'text-align': 'center'}),
                html.Br(),
                dcc.Dropdown(['Afro-Asiatic', 'Arawakan', 'Atlantic-Congo', 'Austronesian', 'Dravidian', 'Eskimo-Aleut', 
                        'Indo-European', 'Mande', 'Pama-Nyungan', 'Sino-Tibetan', 'Tungusic', 'Tupian', 'Turkic', 'Uralic'], 
                        'Afro-Asiatic', id='demo-dropdown'),

                dbc.Row([
                    dbc.Col(id='text3', width=4),
                    dbc.Col(id='text1', width=4),
                    dbc.Col(id='text2', width=4),
                    ], style={"padding-top": "2rem", "padding-bottom": "2rem"}),
                    
                html.Div(children=[
                        dbc.Row([
                            dbc.Col([
                                    html.H5('Состав языковой семьи:', style={'text-align': 'center'}),
                                    dcc.Graph(id='graph2'),
                                    ], width=5),
                    
                            dbc.Col([
                                    html.H5('Средние значения по категориям:', style={'text-align': 'center'}),
                                    dcc.Graph(id='graph1'),
                                    ], width=7),

                            ])
                            
                        ], ),


                html.Div(children=[
                                dbc.Row([
                                    dbc.Col([
                                            html.H5('Самые похожие тренды по категориям:', style={'text-align': 'center'}),
                                            dcc.Graph(id='graph3'),
                                            ]),
                            
                                    dbc.Col([
                                            html.H5('Самые различающиеся тренды по категориям:', style={'text-align': 'center'}),
                                            dcc.Graph(id='graph4'),
                                            ]),

                                    ])
                ]),
                                
            ]),

            

            ], justify="center")
                    
        ], style={"padding": "3rem"})

    app.run_server(debug=True)
