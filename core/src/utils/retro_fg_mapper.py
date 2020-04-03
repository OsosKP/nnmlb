import pandas as pd

df = pd.read_csv('core/data/ChadwickPeople.csv')
df = df[~df['key_fangraphs'].isnull()]
df = df[['key_retro', 'key_fangraphs']]
df['key_fangraphs'] = df['key_fangraphs'].astype(int)
df.rename(columns={'key_retro': 'retroID',
                   'key_fangraphs': 'fangraphsID'}, inplace=True)
df = df.dropna()


def get_fg_id(retro_id):
    return df[df['retroID'] == retro_id]['fangraphsID'].iloc[0] \
        if retro_id in list(df['fangraphsID']) else -1


def get_rs_id(fangraphs_id):
    return df[df['fangraphsID'] == fangraphs_id]['retroID'].iloc[0] \
        if fangraphs_id in list(df['fangraphsID']) else -1
