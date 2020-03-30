import pandas as pd

df = pd.read_csv('../core/output/batters.csv')
df = df[['retroID', 'pos_1B', 'pos_2B', 'pos_3B',
         'pos_C', 'pos_OF', 'pos_P', 'pos_SS']]
df.set_index('retroID', inplace=True)
df = df[df == 1].stack().reset_index().drop(0, 1)
df.rename(columns={'level_1': 'POS'}, inplace=True)
df['POS'] = df['POS'].apply(lambda pos: pos[4:])
pos_dict = df.set_index('retroID').to_dict()['POS']


def get_pos(retroId):
    return pos_dict[retroId] or 'U'
