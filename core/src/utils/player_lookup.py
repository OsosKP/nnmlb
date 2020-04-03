import pandas as pd

df = pd.read_csv('../core/data/Names.csv')


def get_name(retro_id):
    return df[df['retroID'] == retro_id]['Name'].iloc[0]


def get_id(name):
    return df[df['Name'] == name]['retroID'].iloc[0]


def get_by(column, data):
    return df[df[column] == data].iloc[0]
