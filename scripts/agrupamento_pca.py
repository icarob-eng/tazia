import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import os

df = pd.read_pickle(os.path.join('data', 'features_galaxias.pkl'))
csv_classes = pd.read_csv(os.path.join('data', 'gz2_hart16_classes_simple.csv'))

# Converter IDs para texto
csv_classes['dr7objid'] = csv_classes['dr7objid'].astype(str)
if 'dr7objid' in df.columns:
    df['dr7objid'] = df['dr7objid'].astype(str)
else:
    df['dr7objid'] = df['id'].astype(str)

df_completo = pd.merge(df, csv_classes[['dr7objid', 'hubble_class']], on='dr7objid', how='inner')

# tradução:
# E  -> vira 0 (Elípticas)
# S  -> vira 1 (Espirais)
# SB -> vira 1 (Espirais)

mapa_classes = {
    'E': '0 - Elípticas',
    'S': '1 - Espirais',
    'SB': '1 - Espirais'
}
df_completo['classe_binaria'] = df_completo['hubble_class'].map(mapa_classes)

colunas_texto = ['dr7objid', 'hubble_class', 'id', 'asset_id', 'gz2_class', 'classe_binaria']
cols_drop = [c for c in colunas_texto if c in df_completo.columns]

X = df_completo.drop(columns=cols_drop).values

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# K means
kmeans = KMeans(n_clusters=2, random_state=42) 
clusters = kmeans.fit_predict(X)


plt.figure(figsize=(14, 6))

# Gráfico 1: IA (Clusters K-Means)
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='viridis', s=100)
plt.title('IA (K-Means 0 vs 1)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster IA')

# Gráfico 2: (Elípticas vs Espirais)
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df_completo['classe_binaria'], palette='viridis', s=100)
plt.title('Verdade (E vs S+SB)')
plt.xlabel('PCA 1')
plt.legend(title='Classe Real')

plt.tight_layout()
plt.show()
