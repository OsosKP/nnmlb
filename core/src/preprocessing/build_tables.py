from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df_bat = pd.read_csv('core/output/batting_pre.csv')
df_field = pd.read_csv('core/output/fielding_pre.csv')
df_catch = pd.read_csv('core/output/catching_pre.csv')
df_pitch = pd.read_csv('core/output/pitching_pre.csv')
df_meta = pd.read_csv('core/output/metadata.csv')

df_meta.drop(columns=['birthYear'], inplace=True)

df_meta_pos = pd.get_dummies(df_meta['POS'], prefix='pos')
df_meta_bats = pd.get_dummies(df_meta['bats'], drop_first=True, prefix='bats')
df_meta_throws = pd.get_dummies(df_meta['throws'], prefix='throws')

dropped_meta_cols = ['POS', 'bats', 'throws']
df_meta.drop(columns=dropped_meta_cols, inplace=True)

df_meta = df_meta.join([df_meta_pos, df_meta_bats, df_meta_throws])
df_meta.drop(columns=['throws_S', 'throws_R', 'bats_R'], inplace=True)

scaler = MinMaxScaler()

df_meta[['weight', 'height']] = scaler.fit_transform(
    df_meta[['weight', 'height']])

df = pd.merge(df_meta, df_bat, how='inner', on=['retroID'])

df_catch.rename(columns={'CS': 'CS_A', 'SB': 'SB_A'}, inplace=True)
catchers = pd.merge(df_catch, df, how='inner', on=['retroID'])
catchers.drop(columns=['pos_1B', 'pos_2B', 'pos_3B', 'pos_C',
                       'pos_OF', 'pos_P', 'pos_SS'], inplace=True)
unwanted_pitching_columns = ['W', 'L', 'G', 'GS', 'SV']
pitchers = df_pitch.drop(columns=unwanted_pitching_columns)
pitchers['K%'] = pitchers['SO'] / pitchers['BFP']
pitchers['K%'].fillna(0, inplace=True)
fielders = pd.merge(df_field, df, how='inner', on=['retroID'])
fielders = fielders[~fielders['retroID'].isin(pitchers['retroID'])]
fielders = fielders[~(fielders['retroID'].isin(
    catchers['retroID']) & fielders['pos_C'] == 1)]

df.to_csv('core/output/batters.csv', index=False, float_format='%g')
catchers.to_csv('core/output/catchers.csv', index=False, float_format='%g')
fielders.to_csv('core/output/fielders.csv', index=False, float_format='%g')
pitchers.to_csv('core/output/pitchers.csv', index=False, float_format='%g')
