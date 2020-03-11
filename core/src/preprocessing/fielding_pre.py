import pandas as pd
from utils.retroid_dict import get_retroid
pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv(
    'core/data/lahman/mlb_data/Fielding.csv').sort_values('playerID')

df['playerID'] = df['playerID'].apply(get_retroid)
df.rename(columns={'playerID': 'retroID'}, inplace=True)

columns_to_drop = ['stint', 'teamID', 'lgID', 'G']
df.drop(columns=columns_to_drop, inplace=True)
df_catchers = df[df['POS'] == 'C']

df_catchers['GS'].fillna(value=0, inplace=True)
df_catchers['InnOuts'].fillna(value=0, inplace=True)
df_catchers['WP'].fillna(value=0, inplace=True)
df_catchers['SB'].fillna(value=0, inplace=True)
df_catchers['CS'].fillna(value=0, inplace=True)
df_catchers['ZR'].fillna(value=0, inplace=True)

df['GS'].fillna(value=0, inplace=True)
df['InnOuts'].fillna(value=0, inplace=True)
catcher_columns = ['PB', 'WP', 'SB', 'CS', 'ZR']
df.drop(columns=catcher_columns, inplace=True)
df = df[df['POS'] != 'C']
df.drop(columns=['yearID'], inplace=True)
df_catchers.drop(columns=['yearID'], inplace=True)
df['E'].fillna(value=0, inplace=True)
df = df.groupby('retroID').sum().reset_index()
df_catchers = df_catchers.groupby('retroID').sum().reset_index()

df.to_csv('core/output/fielding.csv')
df_catchers.to_csv('core/output/catching.csv')
