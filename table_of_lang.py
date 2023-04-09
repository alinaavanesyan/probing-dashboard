import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from pprint import pprint
import random
import datetime
import time
import sqlite3
import pandas as pd
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import unicodedata
from collections import Counter

ua = UserAgent(verify_ssl=False)
session = requests.session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

lang_file = pd.read_excel('table2.xlsx')
languages = lang_file['Language'].tolist()

def get_article(pglink):
    #заходим в конкретный репозиторий из списка, ищем там conllu-файл, название которого начинается с аббревиатуры, используемой в статье
    req = session.get(f'https://github.com{pglink}', headers={'User-Agent': ua.random})
    soup = BeautifulSoup(req.text, 'html.parser')
          
    if soup is not None:
        conllu_file = soup.find_all('a', {'class': 'js-navigation-open Link--primary'})
        for file in conllu_file:
                name = file['title']
                if name.endswith('.conllu'):
                    index_el = name.index('_')
                    brief = name[0:index_el]
                    return brief
    
def get_page(number):
    #проходимся по списку репозиториев
    req = session.get(f'https://github.com/orgs/UniversalDependencies/repositories?page={number}&type=all', headers={'User-Agent': ua.random})
    soup = BeautifulSoup(req.text, 'html.parser')
    result = {}
    links = soup.find_all('h3', {'class': 'wb-break-all'})
    for link in links:
        link = link.find('a')
        link = link['href']
        permanent ="/UniversalDependencies/UD_"
        new_link = link

        if link.startswith(permanent):
            name = link.replace(permanent,'')
            el = name.index('-')
            name = name[0:el]
            for i in name:
                if i == '_':
                    name = name.replace(i, ' ')
            if name in languages:
                brief = get_article(new_link)
                if brief:
                    result[name] = brief
    return result

results = {}
for x in range(1, 12):
    if results == {}:
        results = get_page(x)
    else:
        res = get_page(x)
        results = results | res
    
results = dict(sorted(results.items()))
sorted_df = lang_file.sort_values(by='Language')
sorted_df['Codes'] = results.values()
sorted_df.set_index('Codes', inplace=True)
sorted_df.to_csv('all_languages.csv', sep='\t', encoding='utf-8')