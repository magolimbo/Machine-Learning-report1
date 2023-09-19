# -*- coding: utf-8 -*-
"""
@Authors
"""

from pandas.plotting import scatter_matrix
import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt

file_path = r"C:\Users\Dell\Desktop\Git\Machine-Learning-report1\Movies_DS.xls"
doc = xlrd.open_workbook(file_path).sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0, 2, 9)

# Extract MPAA names to python list, then encode with integers (dict)
mpaa = doc.col_values(3, 2, 636)
mpaa_name = sorted(set(mpaa))   # set because it deletes the duplicates
mpaaDict = dict(zip(mpaa_name, range(5)))

# Extract GENRE names to python list, then encode with integers (dict)
# the column Genre was moved to this position in excel
genre = doc.col_values(2, 2, 636)
genre_name = sorted(set(genre))
genreDict = dict(zip(genre_name, range(18)))

title = doc.col_values(1, 2, 636)
title_name = sorted(set(title))
titleDict = dict(zip(title_name, range(627)))

# Extract vector y, convert to NumPy array
y_mpaa = np.array([mpaaDict[value] for value in mpaa])
y_genre = np.array([genreDict[value] for value in genre])
y_title = np.array([titleDict[value] for value in title])

# Create a dataframe from the data
data = pd.DataFrame({'MPAA_Rating': y_mpaa, 'genre': y_genre, 'title': y_title, 'Budget': doc.col_values(4, 2, 636),
                     'Gross': doc.col_values(5, 2, 636), 'release_date': doc.col_values(6, 2, 636),
                     'runtime': doc.col_values(7, 2, 636), 'rating': doc.col_values(8, 2, 636), 'rating_count': doc.col_values(9, 2, 636)})


# DATA CLEANING
# Remove duplicates based on the "title" column
data = data.drop_duplicates(subset='title', keep='first')

# Extract X and y from the cleaned dataframe
X = data[['MPAA_Rating', 'genre', 'Budget', 'Gross',
          'release_date', 'runtime', 'rating', 'rating_count']].values
y_mpaa = data['MPAA_Rating'].values
y_genre = data['genre'].values


# Don't know if this is needed
N_mpaa = len(y_mpaa)
N_genre = len(y_genre)
M = len(attributeNames)
C_mpaa = len(mpaa_name)
C_genre = len(genre_name)
