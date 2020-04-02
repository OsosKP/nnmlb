import pandas as pd

df_mlb = pd.read_csv('../core/data/FanGraphsLeagueAverages.csv')
df_al = pd.read_csv('core/data/al_pa.csv')
df_nl = pd.read_csv('core/data/nl_pa.csv')
df_al = df_al['PA']
df_nl = df_nl['PA']
df_mlb.insert(loc=2, column='PA_AL', value=df_al)
df_mlb.insert(loc=2, column='PA_NL', value=df_nl)
df_mlb.to_csv('core/data/FanGraphsLeagueAverages.csv',
              index=False, float_format='%g')
