from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_csv('core/output/pitchers.csv')
df_adv = pd.read_csv('core/output/advanced_pitching.csv')
df = df.drop(columns=['ERA'])
df = df.merge(df_adv, on='retroID', how='left')
df['-ERA'] = 0 - df['ERA']
df['-FIP'] = 0 - df['FIP']
df['Pitching'] = df[['K%', '-ERA', '-FIP', 'WAR']].mean(axis=1).round(3)
scaler = MinMaxScaler()
df['Pitching'] = scaler.fit_transform(df[['Pitching']])
df = df.drop(columns=['-ERA', '-FIP'])

df.to_csv('core/output/pitchers.csv', index=False, float_format='%g')
