import math
import numpy as np
import pandas as pd
from utils.retroid_dict import get_retroid
from utils.team_dict import get_team
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv(
    'core/data/lahman/mlb_data/Batting.csv').sort_values('playerID')
df['playerID'] = df['playerID'].apply(get_retroid)
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


def find_singles(player):
    return player['H'] - player['2B'] - player['3B'] - player['HR']


singles = df.apply(find_singles, axis=1)
df.insert(loc=8, column='1B', value=singles)


def find_pa(player):
    return int(player['AB'] + player['BB'] + player['HBP'] + player['SF'] + player['SH'])


pa_list = df.apply(find_pa, axis=1)
df.insert(loc=6, column='PA', value=pa_list)
df['teamID'] = df['teamID'].apply(get_team)
df = df.sort_index()
df.to_csv('core/output/batting.csv', index=False, float_format='%g')
df.reset_index(inplace=True)

metadata_column_labels = ['index', 'yearID', 'stint', 'teamID']
df.drop(columns=metadata_column_labels, inplace=True)
df = df.groupby('retroID').sum().reset_index()
df['NL'] = np.where(df['NL'] > 0, 1, 0)

df.to_csv('core/output/batting_pre.csv', index=False, float_format='%g')
