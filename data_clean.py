"""
Data processing and data structure establishment
"""
# pandas module: for data cleaning
# numpy module: for data calculation
# re module: for string cleaning
# sk-learn module: for K-means clustering and desision tree estabulishment
# warnings module: for filtering warnings.

import pandas as pd
import numpy as np
import json
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import warnings
warnings.filterwarnings('ignore')

# Attributes construction: max price and min price
def get_price(x):
    # first combine the price list into a string and then delte the ","
    x = ''.join(x).replace(',', '')
    # Use regular expressions to extract numeric information
    x = re.findall('[\d,]+', x)
    if len(x) == 0:
        max_price = -1
        min_price = -1
    elif len(x) == 1:
        max_price = x[0]
        min_price = -1
    else:
        min_price = x[0]
        max_price = x[1]
    # Returns the maximum and minimum prices after cleaning
    return [min_price, max_price]

#  Attributes construction: Whether it is a studio
def is_studio(x):
    # Determine if Studio exists in X. Returns 1 if present, 0 otherwise.
    if "Studio" in x:
        return 1
    else:
        return 0

# Attributes construction: floorplan of the apartment
def get_housetype(x):
    x = re.findall('(\d)-(\d) Bed|(\d) Bed', x)
    if len(x) == 0:
        return np.array([0] * 6)
    else:
        x = x[0]
        x = ''.join(x)
            start = int(x[0])
            end = int(x[1])
            return np.arange(start, end + 1)
        else:
            return np.array([int(x)])

# data "cleaning" function
def DataClean(df1):
    df1['pricing'] = df1['pricing'].apply(get_price)
    df1['min_price'] = df1['pricing'].str[0]
    df1['max_price'] = df1['pricing'].str[1]

    df1['ifStudio'] = df1['beds'].apply(is_studio)

    df1['assist'] = df1['beds'].apply(get_housetype)
    df1['unit_1'] = df1['assist'].apply(lambda x:1 if 1 in x else 0)
    df1['unit_2'] = df1['assist'].apply(lambda x:1 if 2 in x else 0)
    df1['unit_3'] = df1['assist'].apply(lambda x:1 if 3 in x else 0)
    df1['unit_4'] = df1['assist'].apply(lambda x:1 if 4 in x else 0)
    df1['unit_5'] = df1['assist'].apply(lambda x:1 if 5 in x else 0)
    df1['unit_6'] = df1['assist'].apply(lambda x:1 if 6 in x else 0)

    #  Attributes construction: address 
    df1[['street', 'city', 'state']] = df1['sub_title'].str.split(',', expand=True)
    # one hot code
    df_sub = pd.get_dummies(df1[['city', 'state']])
    df1 = pd.concat((df1, df_sub), axis=1)

    # attributes selection
    return df1[['min_price',
           'max_price', 'ifStudio', 'unit_1', 'unit_2', 'unit_3',
           'unit_4', 'unit_5', 'unit_6']]

# user data processing
def data_preproccess(price,beds,if_studio):
    l = get_price(price) 
    if if_studio=='yes': 
        beds += ' Studio'

    ht = get_housetype(beds) .

    df = pd.DataFrame([[0] * 9], columns=['min_price', 'max_price', 'ifStudio',
                                          'unit_1', 'unit_2', 'unit_3',
                                          'unit_4', 'unit_5', 'unit_6'])
    df['min_price'] = l[0]
    df['max_price'] = l[1]
    df['ifStudio'] = is_studio(beds)
    for x in ht:
        if x==1:
            df['unit_1'] = 1
        elif x==2:
            df['unit_2'] = 1
        elif x==3:
            df['unit_3'] = 1
        elif x==4:
            df['unit_4'] = 1
        elif x==5:
            df['unit_5'] = 1
        elif x==6:
            df['unit_6'] = 1

    return df

# main function 
def main(price, beds, if_studio):
    # reading local apartment.json file
    with open('apartment.json', mode='r') as f:
        data = json.load(f)
    # convert the data in to list
    df = pd.DataFrame(data)
    # Place a copy of df. Pass into the data cleansing function and create new_df cleansed data.
    new_df = DataClean(df.copy())
    # Using clustering algorithm label the  
    score = 0
    while score <= 0.5:
        k = KMeans()
        k.fit(new_df)
        y = k.fit_predict(new_df)
        score = silhouette_score(new_df, y)
    df['cls'] = y 
    dtc = DecisionTreeClassifier() # Initialize a decision tree 
    dtc.fit(new_df, y) # Generating
    # desicion tree visualization
    export_graphviz(dtc, out_file='./Tree_entropy.dot', feature_names=new_df.columns)
    user_data = data_preproccess(price, beds, if_studio) # pass the user's three data into a function that processes the user's data
    user_favorite = dtc.predict(user_data) # Use decision trees to predict the user's preference category
    recommend_data = df.loc[df['cls'] == user_favorite[0]] # Output all the category data for the predicted category.
    recommend_data.drop_duplicates(['sub_title'],inplace=True)
    return recommend_data
