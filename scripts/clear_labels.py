import numpy as np
import pandas as pd


df = pd.read_csv('../data/gz2_hart16.csv')

# drop cols
keep = ['dr7objid', 'gz2_class', 'total_classifications']
df.drop(columns=filter(lambda c: c not in keep, list(df.columns)), inplace=True)

# adding asset_id
mapping = pd.read_csv('../data/gz2_filename_mapping.csv')
mapping.drop(columns=['sample'], inplace=True)
mapping.rename(columns={'objid': 'dr7objid'}, inplace=True)
df = df.merge(mapping, on='dr7objid', how='left')
del mapping

# Hubble classification
rules = [
    df.gz2_class.str.contains('E'),
    df.gz2_class.str.contains('S[^B]'),
    df.gz2_class.str.contains('SB'),
]
values = ['E', 'S', 'SB']
df['hubble_class'] = np.select(rules, values, default='?')

print(df.head())
df.to_csv('gz2_hart16_classes.csv', index=False)

# filtering
df.drop(df[df.gz2_class.str.contains('[(].[)]')].index, inplace=True)  # "oddities"
df.drop(df[df.gz2_class.str.contains('A')].index, inplace=True)        # Artifacts
df.drop(df[df.gz2_class.str.contains('Ec')].index, inplace=True)       # Elliptical cigar-chapped
df.drop(df[df.gz2_class.str.contains('Se')].index, inplace=True)       # Spirals edge-on
df.drop(df[df.gz2_class.str.contains('S.?a')].index, inplace=True)     # Spirals edge-on

df.to_csv('gz2_hart16_classes_simple.csv', index=False)
