from scipy.io import loadmat
from matplotlib.pyplot import (figure, subplot, plot, xlabel, ylabel, title,
                               yticks, show, legend, imshow, cm)
import scipy.linalg as linalg
from scipy.linalg import svd
import matplotlib.pyplot as plt
import numpy as np
import xlrd
import pandas as pd
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# -*- coding: utf-8 -*-
"""
NEW ONE

"""

file_path = r"C:\Users\Dell\Desktop\Git\Machine-Learning-report1\Movies_DS.xls"
doc = xlrd.open_workbook(file_path).sheet_by_index(0)

# Extract attribute names
attributeNames = doc.row_values(0, 2, 10)

# Extract MPAA names to python list, then encode with integers (dict)
mpaa = doc.col_values(3, 2, 636)
mpaa_name = sorted(set(mpaa))   # set because it deletes the duplicates
mpaaDict = dict(zip(mpaa_name, range(5)))

# Extract names to python list, then encode with integers (dict)
# the column Genre was moved to this position in excel
genre = doc.col_values(2, 2, 636)
genre_name = sorted(set(genre))
genreDict = dict(zip(genre_name, range(18)))

title = doc.col_values(1, 2, 636)
title_name = sorted(set(title))
titleDict = dict(zip(title_name, range(627)))

rating = doc.col_values(8, 2, 636)
rating_name = sorted(set(rating))
ratDict = dict(zip(rating_name, range(627)))

# Extract vector y, convert to NumPy array
y_mpaa = np.array([mpaaDict[value] for value in mpaa])
y_genre = np.array([genreDict[value] for value in genre])
y_title = np.array([titleDict[value] for value in title])
y_rat = np.array([ratDict[value] for value in rating])

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
y_gross = data['Gross'].values > 4*data['Budget'].values
y_rat = data['rating'].values > 7

N_mpaa = len(y_mpaa)
N_genre = len(y_genre)
M = len(attributeNames)
C_mpaa = len(mpaa_name)
C_genre = len(genre_name)

# Creating dictionaries
genreNames = [gen for gen in genreDict]
mpaaNames = [mpa for mpa in mpaaDict]
ratingNames = [rat for rat in ratDict]

# Select subset of digits classes to be inspected
class_mask = np.zeros(N_mpaa).astype(bool)

# Selection of the genree to visualize
# genres = range(18)
genres = [0, 2, 9, 15]
# mpaas = [1, 3]
# gross = [0, 1]
# ratings = [0, 1, 2, 3]

# =============================================================================
# for v in gross:
#     cmsk = y_gross == v
#     class_mask = class_mask | cmsk
# =============================================================================

# =============================================================================
# for v in ratings:
#     cmsk = y_rat == v
#     class_mask = class_mask | cmsk
# =============================================================================

for v in genres:
    cmsk = y_genre == v
    class_mask = class_mask | cmsk

# =============================================================================
# for v in mpaas:
#     cmsk = y_mpaa == v
#     class_mask = class_mask | cmsk
# =============================================================================
X = X[class_mask, :]
# y_mpaa = y_mpaa[class_mask]
y_genre = y_genre[class_mask]
# y_gross = y_gross[class_mask]
# y_rat = y_rat[class_mask]

N = X.shape[0]

## PCA
Xc = X - np.ones((N, 1))*X.mean(axis=0)
Xc = Xc*(1/np.std(X, 0))
# PCA by computing SVD of Y
U, S, V = svd(Xc, full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum()

# Project data onto principal component space
Z = Xc @ V

# threshold = sum(rho[:4])
threshold = 0.8

# # Create a new figure with 2D projection
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Movies Genres projected on PCs')

# =============================================================================
# # PCs CUMULATIVE PLOT
# # Plot variance explained
# plt.figure()
# plt.plot(range(1, len(rho)+1), rho, 'x-')
# plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
# plt.plot([1, len(rho)], [threshold, threshold], 'k--')
# plt.title('Variance explained by principal components')
# plt.xlabel('Principal component')
# plt.ylabel('Variance explained')
# plt.legend(['Individual', 'Cumulative', 'Threshold'])
# plt.grid()
# plt.show()
#
# print('Ran PCA part')
# =============================================================================

# =============================================================================
# # MPAA
# colors = ['g', '#e67e22', '#8e44ad', 'r']
#
# for c in mpaas:
#     # Select indices belonging to class c
#     class_mask = (y_mpaa == c)
#     ax.scatter(Z[class_mask, 0], Z[class_mask, 1],
#                marker='o', label=mpaaNames[c], s=50,
#                color=colors[c], alpha=0.9)
# plt.legend()
# plt.show()
# =============================================================================

# GENRES
for c in genres:
    # Select indices belonging to class c
    class_mask = (y_genre == c)
    ax.scatter(Z[class_mask, 0], Z[class_mask, 1],
               marker='o', s=50, alpha=0.9, label=genreNames[c])

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')

plt.legend()
# Show the 3D plot
plt.show()


# =============================================================================
# pcs = [0, 1, 2, 3]
# # =============================================================================
# # legendStrs = ['PC'+str(e+1) for e in pcs]
# # c = ['r', 'g', 'b', 'y']
# # bw = .2
# # r = np.arange(1, X.shape[1]+1)
# # for i in pcs:
# #     plt.bar(r+i*bw, V[:, i], width=bw)
# # plt.xticks(r+bw, attributeNames)
# # plt.xlabel('Attributes')
# # plt.ylabel('Component coefficients')
# # plt.legend(legendStrs)
# # plt.grid()
# # plt.title('PCA Component Coefficients')
# # plt.show()
# # =============================================================================
#
# # =============================================================================
# # # PCA COMPONENT COEFFICIENTS
# # # Define colors and legend labels
# # colors = ['#ff5252', '#ffb142', '#34ace0', '#218c74']
# # legend_labels = ['PC' + str(e + 1) for e in pcs]
# #
# # # Set the bar width and create the bar positions
# # bw = 0.2
# # r = np.arange(1, X.shape[1] + 1)
# #
# # # Create a figure and axis
# # plt.figure(figsize=(12, 6))  # Adjust the figure size
# #
# # # Create bar plots for each selected principal component
# # for i, pc in enumerate(range(V.shape[0])):
# #     plt.bar(r + i * bw, V[:, pc], width=bw,
# #             color=colors[i], label=legend_labels[i])
# #
# # # Customize the plot
# # # Rotate x-axis labels for better readability
# # plt.xticks(r + (bw * len(pcs) / 2), attributeNames, rotation=45)
# # plt.xlabel('Attributes')
# # plt.ylabel('Component Coefficients')
# # plt.legend(legend_labels, loc='upper right')  # Position the legend
# # plt.grid(True, linestyle='--', alpha=0.9)  # Add grid lines with transparency
# # plt.title('PCA Component Coefficients')
# #
# # # Add a horizontal line at y=0 for reference
# # plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
# #
# # # Show the plot
# # plt.tight_layout()  # Adjust layout for better spacing
# # plt.show()
# # =============================================================================
#
# # Assuming V contains PCA component coefficients (rows are PCs, columns are attributes)
# # legend_labels should contain labels for the PCs
# # colors can be defined as needed
# colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00",
#           "#FF00FF", "#00FFFF", "#FFA500", "#800080"]
# legend_labels = [att for att in attributeNames]
#
# # Set the bar width and create the bar positions
#
# # Create a figure and axis
# plt.figure(figsize=(30, 8))
#
# # # Create grouped bar plots for each selected attribute
# # for i in range(V.shape[0]):  # Loop through attributes
# #     plt.bar(pcs, pcs[i], color=colors[i], label=legend_labels[i])
#
# positions = np.arange(len(attributeNames))
#
# # Larghezza delle barre
# larghezza = 0.1  # Larghezza delle singole barre
# spaziamento = 0.1  # Spaziamento tra i gruppi di barre
#
# # Creazione del grafico a barre
# plt.bar(positions - spaziamento, V[0, :], larghezza, label='PC1')
# plt.bar(positions, V[:, 1], larghezza, label='PC2')
# plt.bar(positions + spaziamento, V[:, 2], larghezza, label='PC3')
# plt.bar(positions + 2 * spaziamento, V[:, 3], larghezza, label='PC4')
#
# # Customize the plot
# # plt.xticks(r + (bw * (V.shape[0] - 1) / 2), pcs)
# plt.xlabel('PC{i}')
# plt.ylabel('Component Coefficients')
# plt.legend(legend_labels, loc='upper right')
# plt.grid(True, linestyle='--', alpha=0.9)
# plt.title('PCA Component Coefficients')
#
# # Add a horizontal line at y=0 for reference
# plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
#
# # Show the plot
# plt.tight_layout()
# plt.show()
#
#
# # =============================================================================
# # #FIRST 4 PCs SELECTION
# # Proietta i dati nei primi quattro PCs
# num_pcs = 4
# data_projection = np.dot(X, V[:, :num_pcs])
# # data_projection conterrà la proiezione dei dati nei primi quattro PCs
# # =============================================================================
#
#
# # PLOT 3D 4 PCS
# # Crea una figura 3D
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Seleziona tre PCs da visualizzare in 3D (ad esempio, i primi tre)
# pc1, pc2, pc3, pc4 = 0, 1, 2, 3
#
# # Crea il grafico a dispersione 3D
# scatter = ax.scatter(data_projection[:, pc1], data_projection[:, pc2],
#                      data_projection[:, pc3], c=data_projection[:, pc4], cmap='hot')
#
# # Aggiungi etichette per gli assi
# ax.set_xlabel(f'PC{pc1 + 1}')
# ax.set_ylabel(f'PC{pc2 + 1}')
# ax.set_zlabel(f'PC{pc3 + 1}')
#
# # Aggiungi una barra dei colori per il quarto PC
# cbar = plt.colorbar(scatter)
# cbar.set_label(f'PC{pc4 + 1}')
#
# # Mostra il grafico
# plt.title('Data projected in the first 4 PCs')
# plt.show()
# =============================================================================


# =============================================================================
# # PLOT 2D 3 PCS
# # Seleziona due PCs da visualizzare in 2D (ad esempio, i primi due)
# pc1, pc2, pc3, pc4 = 0, 1, 2, 3
#
# # Crea un grafico a dispersione 2D
# plt.figure(figsize=(10, 8))
# plt.scatter(data_projection[:, pc3], data_projection[:,
#             pc1], c=data_projection[:, pc2], cmap='hot')
#
# # Aggiungi etichette per gli assi
# plt.xlabel(f'PC{pc1 + 1}')
# plt.ylabel(f'PC{pc2 + 1}')
#
# # Aggiungi una barra dei colori per il terzo PC
# cbar = plt.colorbar()
# cbar.set_label(f'PC{pc3 + 1}')
#
# # Mostra il grafico
# plt.title('Data projected in the first 3 PCs')
# plt.show()
#
#
# # =============================================================================
# # ## GAUSSIAN PLOT
# # i = 7
# # mu = np.mean(X[:, i])
# # std = np.std(X[:, i])
# # median = np.median(X[:, 3])
# # gaussian_data = np.random.normal(mu, std, len(X[:, i]))
# #
# # plt.hist(X[:, i], bins=50, density=True, alpha=0.6,
# #          color='b', label='Rating_count')
# #
# # # Overlay the Gaussian curve
# # plt.plot(np.sort(gaussian_data), 1 / (std * np.sqrt(2 * np.pi)) *
# #          np.exp(-(np.sort(gaussian_data) - mu)**2 / (2 * std**2)), color='r', label='Gaussian')
# #
# # plt.xlabel('Rating count')
# # plt.ylabel('Frequency')
# # plt.title('Histogram and Gaussian Curve')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
# # =============================================================================
#
#
# # =============================================================================
# # ##BOX PLOT
# # selected_cols = data[['Gross']]
# # plt.figure(figsize=(10, 6))  # Dimensioni del grafico
# # # Crea il box plot
# # plt.boxplot(selected_cols.values, labels=selected_cols.columns)
# #
# # plt.xlabel('')  # Etichetta dell'asse x
# # plt.ylabel('Valori')     # Etichetta dell'asse y
# # plt.title('Box Plot di "Gross"')  # Titolo del grafico
# #
# # plt.show()  # Mostra il grafico
# # =============================================================================
#
#
# # =============================================================================
# # ## COVARIANCE MATRIX
# # plt.figure()  # Added this to make sure that the figure appear
# # # Transform the data into a Pandas dataframe
# # sns.pairplot(data)
# # plt.show()
# # =============================================================================
# # CORRELATION MATRIX
# subset_df = data.iloc[:, 3:]
#
# # Calcola la matrice di correlazione
# correlation_matrix = subset_df.corr()
#
# # Create a heatmap with cell values using Matplotlib
# plt.figure(figsize=(12, 8))
# cax = plt.matshow(correlation_matrix, cmap='Oranges', aspect='auto')
# plt.colorbar(cax, fraction=0.046, pad=0.04)
# plt.title('Correlation matrix')
# plt.xticks(range(len(correlation_matrix.columns)),
#            correlation_matrix.columns, rotation=90)
# plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
# for i in range(len(correlation_matrix.columns)):
#     for j in range(len(correlation_matrix.columns)):
#         plt.text(
#             i, j, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')
# plt.tight_layout()
# plt.show()
# =============================================================================

# =============================================================================
#
# # -*- coding: utf-8 -*-
# """
# DETAILS
#
# """
#
#
# file_path = r"C:\Users\Dell\Desktop\Git\Machine-Learning-report1\Movies_DS.xls"
# doc = xlrd.open_workbook(file_path).sheet_by_index(0)
#
# # Extract attribute names
# attributeNames = doc.row_values(0, 2, 9)
#
# # Extract MPAA names to python list, then encode with integers (dict)
# mpaa = doc.col_values(3, 2, 636)
# mpaa_name = sorted(set(mpaa))   # set because it deletes the duplicates
# mpaaDict = dict(zip(mpaa_name, range(5)))
#
# # Extract GENRE names to python list, then encode with integers (dict)
# # the column Genre was moved to this position in excel
# genre = doc.col_values(2, 2, 636)
# genre_name = sorted(set(genre))
# genreDict = dict(zip(genre_name, range(18)))
#
# title = doc.col_values(1, 2, 636)
# title_name = sorted(set(title))
# titleDict = dict(zip(title_name, range(627)))
#
# # Extract vector y, convert to NumPy array
# y_mpaa = np.array([mpaaDict[value] for value in mpaa])
# y_genre = np.array([genreDict[value] for value in genre])
# y_title = np.array([titleDict[value] for value in title])
#
# # Create a dataframe from the data
# data = pd.DataFrame({'MPAA_Rating': y_mpaa, 'genre': y_genre, 'title': y_title, 'Budget': doc.col_values(4, 2, 636),
#                      'Gross': doc.col_values(5, 2, 636), 'release_date': doc.col_values(6, 2, 636),
#                      'runtime': doc.col_values(7, 2, 636), 'rating': doc.col_values(8, 2, 636), 'rating_count': doc.col_values(9, 2, 636)})
#
#
# # DATA CLEANING
# # Remove duplicates based on the "title" column
# data = data.drop_duplicates(subset='title', keep='first')
#
# # Extract X and y from the cleaned dataframe
# X = data[['MPAA_Rating', 'genre', 'Budget', 'Gross',
#           'release_date', 'runtime', 'rating', 'rating_count']].values
# y_mpaa = data['MPAA_Rating'].values
# y_genre = data['genre'].values
#
#
# # Don't know if this is needed
# N_mpaa = len(y_mpaa)
# N_genre = len(y_genre)
# M = len(attributeNames)
# C_mpaa = len(mpaa_name)
# C_genre = len(genre_name)
#
# # REGRESSION
#
# # Budget vs. Gross Scatterplot
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 2], data['Gross'], alpha=0.5)
# plt.xlabel('Budget')
# plt.ylabel('Gross')
# plt.title('Budget vs. Gross Scatterplot')
# plt.show()
# # Release Date vs. Gross Line Plot
# # =============================================================================
# # plt.figure(figsize=(12, 6))
# # data.groupby('release_date')['Gross'].mean().plot()
# # plt.xlabel('Release Date')
# # plt.ylabel('Average Gross')
# # plt.title('Release Date vs. Average Gross')
# # plt.xticks(rotation=45)
# # plt.show()
# # =============================================================================
# # # Runtime vs. Gross Scatterplot
# # plt.figure(figsize=(8, 6))
# # plt.scatter(X[:, 3], data['runtime'], alpha=0.5)
# # plt.xlabel('Runtime')
# # plt.ylabel('Gross')
# # plt.title('Runtime vs. Gross Scatterplot')
# # plt.show()
# # data vs. runtime Scatterplot
# plt.figure(figsize=(8, 6))
# plt.scatter(X[:, 4], data['Budget'], alpha=0.5)
# plt.xlabel('data')
# plt.ylabel('budget')
# plt.title('Runtime vs. Gross Scatterplot')
# plt.show()
# # Rating vs. Gross Scatterplot
# # plt.figure(figsize=(8, 6))
# # plt.scatter(X[:, 6], data['Budget'], alpha=0.5)
# # plt.xlabel('Rating')
# # plt.ylabel('Gross')
# # plt.title('Runtime vs. Gross Scatterplot')
# # plt.show()
#
#
# # Classification problem
# # The current variables X and y represent a classification problem, in
# # which a machine learning model will use the sepal and petal dimesions
# # (stored in the matrix X) to predict the class (species of Iris, stored in
# # the variable y). A relevant figure for this classification problem could
# # for instance be one that shows how the classes are distributed based on
# # two attributes in matrix X:
# X_c = X.copy()
# y_mpaa_c = y_mpaa.copy()
# y_genre_c = y_genre.copy()
# attributeNames_c = attributeNames.copy()
#
# # =============================================================================
# # i = 1
# # j = 2
# # mpaa_color = ['r', 'g', 'b','p']
# # plt.title('Mpaa rating classification problem')
# # for c in range(len(mpaa_name)):
# #     idx = y_mpaa_c == c
# #     plt.scatter(x=X_c[idx, i],      # values in x-axe
# #                 y=X_c[idx, j],      # values in y-axe
# #                 c=mpaa_color[c],         # color per c in className
# #                 s=50, alpha=0.5,    # s size of markers, alpha transparency
# #                 label=mpaa_name[c])  # label name
# # plt.legend()
# # plt.xlabel(attributeNames_c[i])
# # plt.ylabel(attributeNames_c[j])
# # plt.show()
# # =============================================================================
#
# genre_color = ['red', 'green', 'blue']
# i = 2
# j = 3
# plt.title('Genre classification problem')
# for c in range(len(genre_name[0:3])):
#     idx = y_genre_c == c
#     plt.scatter(x=X_c[idx, i],      # values in x-axe
#                 y=X_c[idx, j],      # values in y-axe
#                 c=genre_color[c],         # color per c in className
#                 s=50, alpha=0.5,    # s size of markers, alpha transparency
#                 label=genre_name[c])  # label name
# plt.legend()
# plt.xlabel(attributeNames_c[i])
# plt.ylabel(attributeNames_c[j])
# plt.show()
#
# =============================================================================
