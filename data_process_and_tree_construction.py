"""
数据清洗,并建立模型
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

# 特征构造: 创建最大价格 和 最小价格
def get_price(x):
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

# 特征创造: 是否为小型公寓
def is_studio(x):
    if "Studio" in x:
        return 1
    else:
        return 0

# 特征创造: 每个公寓的户型
def get_housetype(x):
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

# 数据清洗
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

    # 街区特征构造
    df1[['street', 'city', 'state']] = df1['sub_title'].str.split(',', expand=True)
    df_sub = pd.get_dummies(df1[['city', 'state']])
    df1 = pd.concat((df1, df_sub), axis=1)

    # 特征选择
    return df1[['min_price',
           'max_price', 'ifStudio', 'unit_1', 'unit_2', 'unit_3',
           'unit_4', 'unit_5', 'unit_6']]

# 用户输入数据处理
def data_preproccess(price,beds,if_studio):
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

# 通过数据构建决策树,通过询问用户 想要什么价格区间,几个房间,是否要ifStudio,最后推荐合适的房屋.并且把数据展示出来!
def main(price, beds, if_studio):
    # 读取本地的json文件
    with open('apartment.json', mode='r') as f:
        data = json.load(f)
    # 生成表格数据DataFrame
    df = pd.DataFrame(data)
    # 使用定义的DataClean去做数据清晰
    new_df = DataClean(df.copy())
    # 聚类算法打标签
    score = 0
    while score <= 0.5:
        k = KMeans()
        k.fit(new_df)
        y = k.fit_predict(new_df)
        score = silhouette_score(new_df, y)
    # 把标签添加到这个表格df中
    df['cls'] = y
    # 生成决策树
    dtc = DecisionTreeClassifier()
    dtc.fit(new_df, y) # 生成树
    # 将树保存到本地
    dot_data = export_graphviz(dtc, out_file=None, feature_names=new_df.columns,filled=True, rounded=True, special_characters=True)
    # 写入json文件
    with open('tree.json', 'w') as f:
        json.dump(dot_data, f)

    # 处理用户数据并预测类别
    user_data = data_preproccess(price, beds, if_studio)
    user_favorite = dtc.predict(user_data)
    recommend_data = df.loc[df['cls'] == user_favorite[0]]
    recommend_data.drop_duplicates(['sub_title'],inplace=True)

    return recommend_data

