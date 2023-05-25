# Dashboard

This Dashboard is a probing visualization tool. It can help to interpret a large amount of files with numerical data that resulted from multilingual probing.

To run the program, you need to put the following items in one folder:
- files *all_languages.csv* and *genealogy.csv*,
- folder *assets*,
- folder *Probing results* (it is not in the repository, these are the results of the probing experiments that you have run),
- file *data_launch.py *,
- file *dash_app.py * (the program itself)

First you need to run the file *data_launch.py *, and then *data_app.py *. *data_launch.py* creates a lot of files that are the basis of Dashboard graphs. You do not need to run this file every time, use it only when you have supplemented the probing results with new files.

File *table_of_lang.py* no need to run: this is the file that creates *all_languages.csv*
