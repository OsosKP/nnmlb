from datetime import datetime
import pandas as pd
from utils.retroid_dict import get_retroid

df1 = pd.read_csv('core/data/lahman/mlb_data/Fielding.csv')
df2 = pd.read_csv('core/data/lahman/mlb_data/People.csv')

fielding_columns = ['playerID', 'POS']
df1 = df1[fielding_columns]

people_columns = ['playerID', 'birthYear', 'bats',
                  'throws', 'weight', 'height', 'debut', 'finalGame']
df2 = df2[people_columns]

df2['debut'].fillna(0, inplace=True)
df2['finalGame'].fillna(0, inplace=True)

df1['playerID'] = df1['playerID'].apply(get_retroid)
df2['playerID'] = df2['playerID'].apply(get_retroid)
df1.rename(columns={'playerID': 'retroID'}, inplace=True)
df2.rename(columns={'playerID': 'retroID'}, inplace=True)


def get_debut_year(player):
    birth_year = player['birthYear']
    debut_as_string = str(player['debut'])
    try:
        debut_year = datetime.strptime(debut_as_string, '%m/%d/%y').year
    except ValueError:
        if debut_as_string == '0':
            debut_year = 0
        else:
            debut_year = datetime.strptime(debut_as_string, '%Y-%m-%d').year
    if debut_year - birth_year > 50:
        debut_year = debut_year - 100
    return debut_year


def get_final_year(player):
    debut_year = get_debut_year(player)
    final_as_string = str(player['finalGame'])
    try:
        final_year = datetime.strptime(final_as_string, '%m/%d/%y').year
    except ValueError:
        if final_as_string == '0':
            final_year = 0
        else:
            final_year = datetime.strptime(final_as_string, '%Y-%m-%d').year
    if final_year - debut_year > 50:
        final_year = final_year - 100
    return final_year


df2['debutYear'] = df2.apply(get_debut_year, axis=1)
df2['finalYear'] = df2.apply(get_final_year, axis=1)

df2.drop(columns=['debut', 'finalGame'], inplace=True)

df1 = df1.groupby('retroID').agg(lambda pos: pd.Series.mode(pos)[0])

df1 = df1.reset_index()
df2 = df2[df2['retroID'].notnull()]
df = pd.merge(df1, df2, on='retroID')

df.to_csv('core/output/metadata.csv', index=False)
