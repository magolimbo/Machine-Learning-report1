# -*- coding: utf-8 -*-
"""
DETAILS

"""

from pandas.plotting import scatter_matrix
import pandas as pd
import xlrd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

file_path = r"C:\Denmark\DTU\School\DTU_1\ML, DM\Machine-Learning-report1-main\Movies_DS.xls"
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

# Calculate mean, median, minimum, maximum, standard deviation and interquartile range values for numeric attributes

# Create an empty dictionary for numeric summaries
numeric_summary = {
    'Attribute': [],
    'Mean Value': [],
    'Median Value': [],
    'Standard Deviation': [],
    'Minimum Value': [],
    'Maximum Value': [],
    'Interquartile Range (IQR)': [],
    'Empirical Sample Variance': []
}

# Define the columns for numeric attributes
numeric_cols = ['Budget', 'Gross', 'release_date','runtime', 'rating', 'rating_count']

# Calculate numeric summaries
for col in numeric_cols:
    numeric_summary['Attribute'].append(col)
    numeric_summary['Mean Value'].append(data[col].mean())
    numeric_summary['Median Value'].append(data[col].median())
    numeric_summary['Standard Deviation'].append(data[col].std())
    numeric_summary['Minimum Value'].append(data[col].min())
    numeric_summary['Maximum Value'].append(data[col].max())
    iqr = data[col].quantile(0.75) - data[col].quantile(0.25)
    numeric_summary['Interquartile Range (IQR)'].append(iqr)
    numeric_summary['Empirical Sample Variance'].append(np.var(data[col], ddof=1)) # Empirical sample variance


# Calculate number of categories, mode (most common category), frequency of the mode

# Create an empty dictionary for categorical summaries
categorical_summary = {
    'Attribute': [],
    'Number of Unique Categories': [],
    'Mode (Most Common Category)': [],
    'Frequency of the Mode': [],
}

# Define the columns for categorical attributes
categorical_cols = ['MPAA_Rating','genre']

# Calculate categorical summaries
for col in categorical_cols:
    categorical_summary['Attribute'].append(col)
    num_unique = data[col].nunique()
    categorical_summary['Number of Unique Categories'].append(num_unique)
    
    mode = data[col].mode().iloc[0]
    categorical_summary['Mode (Most Common Category)'].append(mode)
    
    mode_count = (data[col] == mode).sum()
    mode_percentage = (mode_count / len(data)) * 100
    categorical_summary['Frequency of the Mode'].append(f"{mode_count} ({mode_percentage:.2f}%)")

# Create DataFrames for numeric and categorical summaries
numeric_summary_df = pd.DataFrame(numeric_summary)
categorical_summary_df = pd.DataFrame(categorical_summary)

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:,.2f}'.format

# Print or save the summary statistics
print("Numeric Summaries:")
print(numeric_summary_df)
print("\nCategorical Summaries:")
print(categorical_summary_df)

# Reset the display option to its default after printing
pd.reset_option('display.float_format')


# Create subplots for each numeric variable
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    plt.hist(data[col], bins=25, edgecolor='k', alpha=0.7)  # You can adjust the number of bins and other parameters
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# Create subplots for each categorical variable
# Mpaa_rating
plt.figure(figsize=(8, 6))
data[categorical_cols[0]].value_counts().plot(kind='pie', labels=list(mpaaDict.keys()), autopct='%1.1f%%')
plt.title(f'Pie Chart of {categorical_cols[0]}')
plt.xlabel('MPAA rating')
plt.ylabel('')
plt.show()

# Genres
plt.figure(figsize=(8, 6))
ax = data[categorical_cols[1]].value_counts().plot(kind='bar')
plt.title(f'Bar Chart of {categorical_cols[1]}')
plt.xlabel('Genres')
plt.ylabel('Frequency')
ax.set_xticklabels(labels=list(genreDict.keys()), rotation=60)
plt.show()