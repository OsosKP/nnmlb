import pandas as pd
from utils.retroid_dict import get_retroid

df1 = pd.read_csv('../core/data/lahman/mlb_data/Fielding.csv')
df2 = pd.read_csv('../core/data/lahman/mlb_data/People.csv')

fielding_columns = ['playerID', 'POS']
df1 = df1[fielding_columns]

people_columns = ['playerID', 'birthYear',
                  'bats', 'throws', 'weight', 'height']
df2 = df2[people_columns]

df1['playerID'] = df1['playerID'].apply(get_retroid)
df2['playerID'] = df2['playerID'].apply(get_retroid)
df1.rename(columns={'playerID': 'retroID'}, inplace=True)
df2.rename(columns={'playerID': 'retroID'}, inplace=True)

df1 = df1.groupby('retroID').agg(pd.Series.mode)

df1 = df1.reset_index()
df2 = df2[df2['retroID'].notnull()]
df = pd.merge(df1, df2, on='retroID')

df.to_csv('output/metadata.csv')
