# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:44:15 2023

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt
import xlrd
from sklearn import decomposition
import pandas as pd


doc = xlrd.open_workbook(
    r"C:\Users\Dell\Desktop\Git\Machine-Learning-report1\Movies_DS.xls").sheet_by_index(0)

# extract attributes/features names
attributeNames = doc.row_values(0, 0, 10)
print(attributeNames)
dtypes = [('movieID', int), ('title', '<U26')]
# dtypes = [('movieID', int), ('title', '<U26'), ('MPAA_Rating', '<U26'), ('Budget', float), ('Gross', float),
#           ('release_date', '<U26'), ('genre', '<U26'), ('runtime', int), ('rating', float), ('rating_count', int)]

# types= [int,'<U26','<U26',float,]

# extract movie names
moviesNames = doc.col_values(1, 1, 636)
# with set() all the rows with the same movie title feature are removed
classLabelsnodups = set(moviesNames)
print("the original dataset has " +
      str(len(moviesNames) - len(classLabelsnodups)) + " duplicates")

X = np.empty((635,), dtype=dtypes)

X['movieID'] = range(635)
X['title'] = np.asarray(moviesNames, dtype='<U26')


# for i, col_id in enumerate(range(4, 5)):
#     X[:, i] = np.array(doc.col_values(col_id, 1, 637))

# matrice 635 x 8 colonne
# X = np.empty((len(moviesNames), len(attributeNames)))
# X[:, 0] = classLabels

# Data attributes to be plotted
# gross = 4

# plt.figure(1)
# plt.plot(moviesNames, X[:, gross], 'o')
# plt.show()
# ##
# # Make a simple plot of the i'th attribute against the j'th attribute
# # Notice that X is of matrix type (but it will also work with a numpy array)
# X = np.array(X)  # Try to uncomment this line


# # Make another more fancy plot that includes legend, class labels,
# # attribute names, and a title.

# plt.title('NanoNose data')

# plt.xlabel("Budget")
# plt.ylabel("Gross")

# # Output result to screen
# plt.show()
