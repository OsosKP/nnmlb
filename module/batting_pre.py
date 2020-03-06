import math
import numpy as np
import pandas as pd
from team_dict import team_dict

# We're going to be reassigning columns, so we'll turn off this warning - we know what we're doing!
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('data/lahman/mlb_data/Batting.csv').sort_values('playerID')
df.rename(columns={'playerID': 'retroID'}, inplace=True)
df['IBB'].fillna(value=0, inplace=True)
df['SF'].fillna(value=0, inplace=True)
df_temp = df[(df['CS'].notnull())]
total_sb = df_temp['SB'].sum()
total_cs = df_temp['CS'].sum()
df_temp['CS'] = df_temp.apply(lambda x: x['SB'] / 2, axis=1)


def fill_cs(data):
    return data['SB'] / 2 if math.isnan(data['CS']) else data['CS']


df['CS'] = df.apply(fill_cs, axis=1)
df['GIDP'].fillna(value=0, inplace=True)
df['NL'] = pd.get_dummies(df['lgID'], drop_first=True)
df.drop(columns=['lgID'], inplace=True)

df['teamID'] = df['teamID'].apply(lambda x: team_dict()[x])
df = df.sort_index()
df.reset_index(inplace=True)

metadata_column_labels = ['index', 'yearID', 'stint', 'teamID']
metadata = df[metadata_column_labels].set_index(df['retroID']).reset_index()
indexer = metadata.drop_duplicates('retroID').set_index(
    'index').T.to_dict('retroID')[0]
df = df.drop(columns=metadata_column_labels)
df = df.groupby('retroID').sum().reset_index()
df['NL'] = np.where(df['NL'] > 0, 1, 0)
tensor = df.drop(columns=['retroID'])

tensor.to_csv('./output/tensor.csv')
