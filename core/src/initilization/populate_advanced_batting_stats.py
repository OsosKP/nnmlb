import pandas as pd
from utils.retro_fg_mapper import get_rs_id

df_names = pd.read_csv('core/data/Names.csv')
df_stats = pd.read_csv('core/data/FanGraphsAllStats.csv')

df_stats['retroID'] = df_stats.apply(
    lambda player: get_rs_id(player['playerid']), axis=1)

df_stats = df_stats[df_stats['retroID'] != -1]
df_stats = df_stats[['retroID', 'wOBA', 'wRC+', 'WAR']]

df_stats.to_csv('core/output/advanced_batting.csv',
                index=False, float_format='%g')
