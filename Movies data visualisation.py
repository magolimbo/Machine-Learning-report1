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
    r'C:\Users\Dell\OneDrive - Danmarks Tekniske Universitet\Machine learning Fall23\Movies_DS.xls').sheet_by_index(0)

# extract attributes/features names
attributeNames = doc.row_values(0, 0, 8)
print(attributeNames)

# extract movie names
classLabels = doc.col_values(1, 1, 636)
# with set() all the rows with the same movie title feature are removed
classLabelsnodups = set(classLabels)
print("the original dataset has " +
      str(len(classLabels) - len(classLabelsnodups)) + " duplicates")

# matrice 635 x 8 colonne
X = np.empty((len(classLabels), len(attributeNames)))
# X[:, 0] = classLabels


# # MOVIES DOC

# d = 3
# e = 4
# final_df = final_df.drop_duplicates(subset='MovieID')
# plt.figure(1)
# plt.plot(final_df.iloc[:, 7], final_df.iloc[:, e], 'o', color='r', alpha=0.5)
# plt.xlabel("Rating")
# plt.ylabel("Gross")
# plt.show()


# # Select the second and third columns from the DataFrame and convert
# # them to a NumPy array
# final_df.iloc[:, [2, 3]] = final_df.iloc[:, [
#     2, 3]].apply(pd.to_numeric, errors='coerce')

# data_subset = final_df.iloc[:, [3, 4]].values


# # Calculate the covariance matrix
# cov_movies = np.cov(data_subset, rowvar=False, ddof=1,
#                     fweights=None, aweights=None)

# # Print the covariance matrix
# print(cov_movies)
