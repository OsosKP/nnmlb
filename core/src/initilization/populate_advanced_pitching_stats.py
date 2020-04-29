import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils.retro_fg_mapper import get_rs_id

df_names = pd.read_csv('core/data/Names.csv')
df_stats = pd.read_csv('core/data/FanGraphsPitching.csv')

df_stats = df_stats.drop(
    columns=['Team', 'W', 'L', 'SV', 'G', 'GS', 'GB%', 'HR/FB', 'xFIP'])

df_stats['retroID'] = df_stats.apply(
    lambda player: get_rs_id(player['playerid']), axis=1)

df_stats.loc[(df_stats['retroID'] == -1), 'retroID'] = 'morrb103'

df_stats = df_stats[['retroID', 'IP', 'K/9', 'BB/9', 'HR/9',
                     'BABIP', 'LOB%', 'ERA', 'FIP', 'WAR']]

scaler = MinMaxScaler()
df_stats['IP'] = scaler.fit_transform(df_stats[['IP']])


def convert_lob_to_float(string):
    val = string[:-2]
    return float(val)


df_stats['LOB%'] = df_stats['LOB%'].apply(convert_lob_to_float)


df_stats.to_csv('core/output/advanced_pitching.csv',
                index=False, float_format='%g')
