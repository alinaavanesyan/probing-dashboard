# Probing Dashboard

This Dashboard is a probing visualization tool. It can help to interpret a large amount of files with numerical data that resulted from multilingual probing.

To run the program, you need to put the following items in one folder:
- files *all_languages.csv* and *genealogy.csv*,
- folder *assets*,
- folder *Probing results* (it is not in the repository, these are the results of the probing experiments that you have run),
- file *data_launch.py*,
- file *dash_app.p * (the program itself)

First you need to run the file *data_launch.py*, and then *data_app.py*. *data_launch.py* creates a lot of files that are the basis of Dashboard graphs. You do not need to run this file every time, use it only when you have supplemented the probing results with new files.

File *table_of_lang.py* no need to run: this is the file that creates *all_languages.csv*

### About the Dashboard
The dashboard includes more than 10 charts. Various types of visualization are used: line graphs, boxplots, heatmaps, horizontal bar chart, scatter plots, treemap and even the combination of types of graphs. All of them are suitable for specific goals:
– heatmaps are good at showing the relationships between dependent variables due to the colored cells and gradient color scale, so, there is no difference how many variables, it doesn’t interfere with perception, which makes heatmap indispensable for big data representation;
– line graphs let us overlay multiple trends on one figure, where the trends can be independent or can demonstrate intersections;
– boxplots are aimed to reflect the spectrum of values corresponding to the accuracy obtained as a result of probing experiments, and etc.

**Why is it needed?** To visualize a massive data resulting from multilingual probing and conduct the analysis of language models. 

