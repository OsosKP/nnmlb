import pandas as pd

df_mlb = pd.read_csv('core/data/FanGraphsLeagueAverages.csv')
df_al = pd.read_csv('core/data/al.csv')
df_nl = pd.read_csv('core/data/nl.csv')
df_al = df_al['wRC']
df_nl = df_nl['wRC']
df_mlb.insert(loc=18, column='wRC_AL', value=df_al)
df_mlb.insert(loc=18, column='wRC_NL', value=df_nl)
df_mlb.to_csv('core/data/FanGraphsLeagueAverages.csv',
              index=False, float_format='%g')
