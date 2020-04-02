import pandas as pd

df_by_player = pd.read_csv('core/output/batters.csv')
df_meta = pd.read_csv('core/data/Lahman/mlb_data/People.csv')
fg_stats = pd.read_csv('core/data/FanGraphsAllStats.csv')

df_meta['Name'] = df_meta['nameFirst'] + ' ' + df_meta['nameLast']

df_names = df_meta[['retroID', 'Name']]

df_names = df_names.dropna()

df_names = df_names.drop_duplicates('retroID', keep='first')

df_names.to_csv('core/data/Names.csv')
