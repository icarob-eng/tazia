import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import os

encoder = 'zoobot-encoder-convnext_nano'
try:
    os.makedirs(f'results/{encoder}')
except FileExistsError:
    pass
features = pd.read_csv(f'data/{encoder}.feature_vector.csv', sep=' ',
                       header=None, names=['asset_id', *range(640)])
features.dropna(axis=0, how='any', inplace=True)
df = pd.read_csv('data/gz2_hart16_classes_simple.csv')

df_features = pd.merge(features, df, on='asset_id', how='inner')  # automatically filters
del features, df

print(df_features.head())

# ------ PCA ------
ffeatures = df_features.drop(columns=[c for c in df_features.columns if type(c) != int])

print(f'Performing pca for {ffeatures.shape=}')

pca = PCA(n_components=2)
pca_results = pca.fit_transform(ffeatures)

df_features['pca_x'] = pca_results[:, 0]
df_features['pca_y'] = pca_results[:, 1]

del pca_results

# ------ k-means ------
print('Performing k-means')
kmeans = KMeans(n_clusters=2, random_state=42)
df_features['cluster'] = kmeans.fit_predict(ffeatures)

# ------ plotting ------
group_labels = df_features.groupby('hubble_class')
group_clusters = df_features.groupby('cluster')

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

fig.suptitle(f'k-means with PCA\n{encoder}')

ax0.set_xlabel('PCA x')
ax0.set_ylabel('PCA y')
ax1.set_xlabel('PCA x')

ax0.set_title('Labels from GalaxyZoo')
for name, group in group_labels:
    ax0.scatter(group.pca_x, group.pca_y, label=name)
ax0.legend()

ax1.set_title('Clusters from k-means')
for name, group in group_clusters:
    ax1.scatter(group.pca_x, group.pca_y, label=name)
ax1.legend()

fig.savefig(f'results/{encoder}/pca-kmeans.png')

# ------ result report ------

mapa = {'E': 0, 'S': 1, 'SB': 1}
df_features['classe_binaria'] = df_features['hubble_class'].map(mapa)

cm = confusion_matrix(
        y_true=df_features['classe_binaria'],
        y_pred=df_features['cluster'],
        labels=df_features['classe_binaria'].unique()
)

cm_df = pd.DataFrame(
    cm,
    index=pd.Index(df_features['classe_binaria'].unique(), name='classe'),
    columns=pd.Index(df_features['cluster'].unique(), name='cluster')
)

result = f'''# Result report

Methodology: k-means grouping with PCA reduction for {encoder}.
Number of samples: {len(df_features)}.

Confusion matrix:
{cm_df}

Normalized:
{cm_df.div(cm_df.sum(axis=1), axis=0)}
'''

print(result)
with open(f'results/{encoder}/pca-kmeans.report.txt', 'w+') as f:
    f.write(result)
