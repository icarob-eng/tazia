import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os

df = pd.read_pickle(os.path.join('data', 'features_galaxias.pkl'))
csv_classes = pd.read_csv(os.path.join('data', 'gz2_hart16_classes_simple.csv'))

# Padronizar IDs como texto
csv_classes['dr7objid'] = csv_classes['dr7objid'].astype(str)
if 'dr7objid' in df.columns:
    df['dr7objid'] = df['dr7objid'].astype(str)
else:
    df['dr7objid'] = df['id'].astype(str)


df_completo = pd.merge(df, csv_classes[['dr7objid', 'hubble_class']], on='dr7objid', how='inner')

# Simplificar Classes (0 = Elíptica, 1 = Espiral)
mapa = {'E': '0 - Elípticas', 'S': '1 - Espirais', 'SB': '1 - Espirais'}
df_completo['classe_binaria'] = df_completo['hubble_class'].map(mapa)

# Se tiver menos que 1000, pega tudo.
qtd = min(1000, len(df_completo))
df_sample = df_completo.sample(n=qtd, random_state=42)

colunas_texto = ['dr7objid', 'hubble_class', 'id', 'asset_id', 'gz2_class', 'classe_binaria']
cols_drop = [c for c in colunas_texto if c in df_sample.columns]
X = df_sample.drop(columns=cols_drop).values

# calcular t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# kmeans
print("🤖 IA tentando agrupar sozinha...")
kmeans = KMeans(n_clusters=2, random_state=42)
clusters_ia = kmeans.fit_predict(X) 

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Gráfico 1: IA
sns.scatterplot(
    x=X_tsne[:,0], y=X_tsne[:,1], 
    hue=clusters_ia, 
    palette='viridis', 
    s=60, ax=axes[0]
)
axes[0].set_title('IA agrupou)')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')

# Gráfico 2: A Verdade
sns.scatterplot(
    x=X_tsne[:,0], y=X_tsne[:,1], 
    hue=df_sample['classe_binaria'], 
    palette='viridis', 
    s=60, ax=axes[1]
)
axes[1].set_title('Verdade')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()