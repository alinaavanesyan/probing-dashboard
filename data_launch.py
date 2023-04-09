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
import os

hits = glob('Probing_framework/results/*/*/*.json', recursive=True)
lang_file = pd.read_csv('all_languages.csv', delimiter=';')
if not os.path.exists('data'):
    os.makedirs('data')

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
with open('data/all_categories.json', 'w', encoding='utf-8') as f:
    json.dump(all_categories, f, ensure_ascii=False, indent=4)
with open('data/middle_all_layers_family.json', 'w', encoding='utf-8') as f:
    json.dump(middle_all_layers_family, f, ensure_ascii=False, indent=4)
with open('data/all_layers_lang.json', 'w', encoding='utf-8') as f:
    json.dump(all_layers_lang, f, ensure_ascii=False, indent=4)
with open('data/lang_files.json', 'w', encoding='utf-8') as f:
    json.dump(lang_files, f, ensure_ascii=False, indent=4)
with open('data/model_name.txt', 'w') as f:
    for name in model_names:
        f.write(str(name) + '\n')
        
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
with open('data/middle_values.json', 'w', encoding='utf-8') as f:
    json.dump(middle_values, f, ensure_ascii=False, indent=4)

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
with open('data/size.json', 'w', encoding='utf-8') as f:
    json.dump(size, f, ensure_ascii=False, indent=4)

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
with open('data/structure.json', 'w', encoding='utf-8') as f:
    json.dump(structure, f, ensure_ascii=False, indent=4)

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
with open('data/cat_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(cat_statistics, f, ensure_ascii=False, indent=4)

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
with open('data/cat_statistics_for_table.json', 'w', encoding='utf-8') as f:
    json.dump(cat_statistics_for_table, f, ensure_ascii=False, indent=4)

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
with open('data/all_layers_lang_middle.json', 'w', encoding='utf-8') as f:
    json.dump(all_layers_lang_middle, f, ensure_ascii=False, indent=4)


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
with open('data/full_layers.json', 'w', encoding='utf-8') as f:
    json.dump(full_layers, f, ensure_ascii=False, indent=4)

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
df_full_layers.to_csv('data/df_full_layers.csv')

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

for model_name in model_names:
    model_name_new =  model_name.replace('/', '%')
    for family in structure[model_name].keys():
        df_info_for_graphs = pd.DataFrame(columns = ['Language', 'Category', 'X', 'Y'])
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
        if family != '[Basque]*':
            path = f'data/graphs/{model_name_new}/{family}'
        else:
            path = f'data/graphs/{model_name_new}/{family[:-1]}'
        if not os.path.exists(path):
            os.makedirs(path)
        df_info_for_graphs.to_csv(f'{path}/df_info_for_graphs.csv')
        for category in df_info_for_graphs['Category'].unique().tolist():
            path_with_cat = f'{path}/{category}'
            if not os.path.exists(path_with_cat):
                os.makedirs(path_with_cat)
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
                with open(f'{path_with_cat}/distances1.json', 'w', encoding='utf-8') as f:
                    json.dump(distances1, f, ensure_ascii=False, indent=4)
                with open(f'{path_with_cat}/distances2.json', 'w', encoding='utf-8') as f:
                    json.dump(distances2, f, ensure_ascii=False, indent=4)
            else:
                df_temp = df_info_for_graphs[df_info_for_graphs['Category'].isin([category])]
                frames = [df_incomparable_graph3, df_temp]
                df_incomparable_graph3 = pd.concat(frames)
                df_incomparable_graph3.to_csv(f'{path_with_cat}/df_incomparable_graph3.csv')

middle_values_lang = {}
for model_name in all_layers_lang.keys():
    middle_values_lang[model_name] = {}
    for language in all_layers_lang[model_name].keys():
        middle_values_lang[model_name][language] = {'f1': {}, 'accuracy': {}}
        for category in all_layers_lang[model_name][language]['f1'].keys():
            value = sum(all_layers_lang[model_name][language]['f1'][category].values())/len(all_layers_lang[model_name][language]['f1'][category].values())
            middle_values_lang[model_name][language]['f1'][category] = round(value,3)
with open('data/middle_values_lang.json', 'w', encoding='utf-8') as f:
    json.dump(middle_values_lang, f, ensure_ascii=False, indent=4)

datasets = {}
for file_name in hits: 
    file = open(file_name)
    data_file = json.loads(file.read())
    lang = data_file['params']['task_language']
    model_name = data_file['params']['hf_model_name']
    if model_name not in datasets.keys():
        datasets[model_name] = {}
    a = lang_file.loc[lang_file['Codes'].isin([lang])]
    lang_full = a.iloc[0]['Language']
    cat = data_file['params']['task_category']
    if lang_full not in datasets[model_name].keys():
        datasets[model_name][lang_full] = {}
    datasets[model_name][lang_full][cat] = {}
    datasets[model_name][lang_full][cat]['training'] = data_file['params']['original_classes_ratio']['tr']
    datasets[model_name][lang_full][cat]['validation'] = data_file['params']['original_classes_ratio']['va']
    datasets[model_name][lang_full][cat]['test'] = data_file['params']['original_classes_ratio']['te']
with open('data/datasets.json', 'w', encoding='utf-8') as f:
    json.dump(datasets, f, ensure_ascii=False, indent=4)

boxplot = pd.DataFrame(columns = ['Model name', 'Family', 'Category', 'Language', 'Average value'])
boxplot_info = {}
for model_name in cat_statistics.keys():
    for family in cat_statistics[model_name]:
        for category in cat_statistics[model_name][family].keys():
            boxplot_info[category] = cat_statistics[model_name][family][category]
            model_names = [model_name]*len(cat_statistics[model_name][family][category].keys())
            families = [family]*len(cat_statistics[model_name][family][category].keys())
            categories = [category]*len(cat_statistics[model_name][family][category].keys())
            languages = list(cat_statistics[model_name][family][category].keys())
            values = list(cat_statistics[model_name][family][category].values())
            df_temp = pd.DataFrame(columns = ['Model name', 'Family', 'Category', 'Language', 'Average value'])
            df_temp['Model name'] = model_names; df_temp['Family'] = families; df_temp['Category'] = categories; 
            df_temp['Language'] = languages; df_temp['Average value'] = values
            frames = [boxplot, df_temp]
            boxplot = pd.concat(frames)

boxplot.reset_index(drop=True)
boxplot.to_csv('data/boxplot.csv')

middle_for_each_cat = {}
for model_name in all_categories:
    middle_for_each_cat[model_name] = {}
    for family in all_categories[model_name]:
        for category in all_categories[model_name][family]['f1']:
            if category not in middle_for_each_cat[model_name]:
                middle_for_each_cat[model_name][category] = []
                middle_for_each_cat[model_name][category].append(1)
                middle_for_each_cat[model_name][category].append(all_categories[model_name][family]['f1'][category])
            else:
                middle_for_each_cat[model_name][category][0] += 1
                middle_for_each_cat[model_name][category][1] += all_categories[model_name][family]['f1'][category]

for model_name in middle_for_each_cat.keys():
    for key in middle_for_each_cat[model_name].keys():
        middle_for_each_cat[model_name][key] = round(middle_for_each_cat[model_name][key][1] / middle_for_each_cat[model_name][key][0], 3)
with open('data/middle_for_each_cat.json', 'w', encoding='utf-8') as f:
    json.dump(middle_for_each_cat, f, ensure_ascii=False, indent=4)