"""
data processing and tree construction
"""
import pandas as pd
import numpy as np
import json
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import warnings
warnings.filterwarnings('ignore')

# Getting the max pirce and min price
def get_price(x):
    ''' Convert the string containing price information into a list for future use
     
    Parameters
    ----------
    x: string
        The string contains the price information

    Returns
    -------
    max_price: string
        The max price of the apartment
        
    min_price: string
        The min price of the apartment

    '''
    x = ''.join(x).replace(',', '')
    x = re.findall('[\d,]+', x)
    if len(x) == 0:
        max_price = -1
        min_price = -1
    elif len(x) == 1:
        max_price = x[0]
        min_price = x[0]
    else:
        min_price = x[0]
        max_price = x[1]
    return [min_price, max_price]

# Testify whether there's a studio 
def is_studio(x):
    ''' Testify if there's a studio floorplan
     
    Parameters
    ----------
    x: string
        The string contains the floorplan information

    Returns
    -------
    0
        No studio in the floor plan
        
    1
        There's studio in the floor plan
    '''    
    if "Studio" in x:
        return 1
    else:
        return 0

# Getting the floor plan of the apartment

def get_housetype(x):
    ''' Getting the floorplan (how many beds in the apartment) information of the apartment
    If there's no floorplan information, return a zero array. If there's multiple floor plan return a 
    arithmetic sequence start with min number of beds and end with max num of beds. If there's only one 
    floor plan, return a int number indicating the number of beds
    
    Parameters
    ----------
    x: string
        The string contains the floorplan information
        
    start: int
        The minimum number of beds the apartments provides
        
    end: int
        The maximum number of beds the apartments provides

    Returns
    -------
    array([0] * 6)
        reutrn a 0 array if there's no floorplan information
        
    arange(start, end + 1)
        return a arithmetic sequence start with min number of beds and end with max num of beds.
        
    int
        Return a int if there onlt one floor plan
    '''     
    x = re.findall('(\d)-(\d) Bed|(\d) Bed', x)
    if len(x) == 0:
        return np.array([0] * 6)
    else:
        x = x[0]
        x = ''.join(x)
        if len(x) == 2:
            start = int(x[0])
            end = int(x[1])
            return np.arange(start, end + 1)
        else:
            return np.array([int(x)])

# data cleansing
def DataClean(df1):
    ''' Convert the cached data into a more standardized DataFrame format
    Return the apartment pricing, min price, max price, if there's a studio
    and floorplan information (using one-hot code) in Dataframe way

    Returns
    -------
    df1[['min_price','max_price', 'ifStudio', 'unit_1', 'unit_2', 'unit_3','unit_4', 'unit_5', 'unit_6']]: DataFrame
         Return the apartment pricing, min price, max price, if there's a studio and floorplan information in Dataframe way
           
    '''     
    
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

    df1[['street', 'city', 'state']] = df1['sub_title'].str.split(',', expand=True)
    df_sub = pd.get_dummies(df1[['city', 'state']])
    df1 = pd.concat((df1, df_sub), axis=1)

    return df1[['min_price',
           'max_price', 'ifStudio', 'unit_1', 'unit_2', 'unit_3',
           'unit_4', 'unit_5', 'unit_6']]

# processing the user input data
def data_preproccess(price,beds,if_studio):
    ''' Convert the user data into a more standardized DataFrame format
    Return the apartment pricing, min price, max price, if there's a studio
    and floorplan information (using one-hot code) in Dataframe way
    Parameters
    ----------
    x: string
        The string contains the floorplan information
        
    start: int
        The minimum number of beds the apartments provides
        
    end: int
        The maximum number of beds the apartments provides
    Returns
    -------
    df: DataFrame
         Return the apartment pricing, min price, max price, if there's a studio and floorplan information in Dataframe way
           
    '''
    l = get_price(price)
    if if_studio=='yes':
        beds += ' Studio'

    ht = get_housetype(beds)

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


def main(price, beds, if_studio):
    ''' Main function for data process and tree construction
    First read the local json file and convert and process the json file into a standardized Dataframe format
    Then utilize K-means clustering and decisiontreeClassifier to constrcut the tree. Then also convert the user input into a
    standardized dataframe file and feed it into the tree to get the recommendation results
    
    Parameters
    ----------
    data: json
        The cached apartment information
        
    df: Dataframe
        converted aparmtent info from data
        
    new_df: Dataframe
        converted aparmtent info from data after data cleansing
        
    score: float
        A critia to decide when to stop k-means algrithom
    
    dtc: Decisiontree
        The decision tree for the recommendation system
        
    user_data: DataFrame
        The standlized users' information on apartments

    user_favorite: array
        The prediction of the "leaf" in the tree that user would like most 
        
    Returns
    -------
    recommend_data: DataFrame
         Return the recommendation information based on users' answer
           
    '''
    with open('apartment.json', mode='r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    new_df = DataClean(df.copy())
    # K-means clustering
    score = 0
    while score <= 0.5:
        k = KMeans()
        k.fit(new_df)
        y = k.fit_predict(new_df)
        score = silhouette_score(new_df, y)
    # Adding the label into df
    df['cls'] = y
    # Genrating decision tree
    dtc = DecisionTreeClassifier()
    dtc.fit(new_df, y) 
    # save the tree to local
    dot_data = export_graphviz(dtc, out_file=None, feature_names=new_df.columns,filled=True, rounded=True, special_characters=True)
    with open('tree.json', 'w') as f:
        json.dump(dot_data, f)

    # predict users class and return recommendation accordingly
    user_data = data_preproccess(price, beds, if_studio)
    user_favorite = dtc.predict(user_data)
    recommend_data = df.loc[df['cls'] == user_favorite[0]]
    recommend_data.drop_duplicates(['sub_title'],inplace=True)

    return recommend_data

