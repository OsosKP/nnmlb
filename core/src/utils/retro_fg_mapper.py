import pandas as pd

df = pd.read_csv('core/output/retro_fg_ids.csv')


def get_fg_id(retro_id):
    return df[df['retroID'] == retro_id]['fangraphsID'].iloc[0] \
        if retro_id in list(df['fangraphsID']) else -1


def get_rs_id(fangraphs_id):
    return df[df['fangraphsID'] == fangraphs_id]['retroID'].iloc[0] \
        if fangraphs_id in list(df['fangraphsID']) else -1
