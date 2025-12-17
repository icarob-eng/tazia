import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import os



# ------ data ------
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
del df, features

mapa = {'E': 0, 'S': 1, 'SB': 1}
df_features['classe_binaria'] = df_features['hubble_class'].map(mapa)
df_features['not_classe'] = (df_features['classe_binaria'] - 1).abs()  # negation

print(f'Number of E: {df_features['not_classe'].sum()}, S/SB: {df_features['classe_binaria'].sum()}')

X = torch.tensor(df_features[[*range(0, 640)]].to_numpy(), dtype=torch.float32)
y = torch.tensor(df_features[['not_classe', 'classe_binaria']].to_numpy(), dtype=torch.float32)
# [1,0] -> E
# [0,1] -> S/SB

# dividing between train and test
N = X.shape[0]
N_train = int(0.8 * N)

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]

print(f'Total samples: {N}. Train samples: {N_train}. Test samples: {N - N_train}')

# ------ classifier ------
model = nn.Linear(640, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
train_loss_history = np.empty(num_epochs)

for epoch in tqdm.tqdm(range(num_epochs)):
    optimizer.zero_grad()

    logits = model(X_train)
    loss = criterion(logits, y_train)

    loss.backward()
    train_loss_history[epoch] = loss.item()
    optimizer.step()

plt.title('Model training loss')
plt.plot(train_loss_history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(f'results/{encoder}/classifier-loss.png')

model.eval()
with torch.no_grad():
    logits = model(X_test)
    y_pred = torch.argmax(logits, dim=1)


# ------ PCA classes plot ------

pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_test)


df_test = pd.DataFrame({
    'pca_x': pca_results[:, 0],
    'pca_y': pca_results[:, 1],
    'hubble_class': y_test[:, 0],
    'pred_class': y_pred
})
group_labels = df_test.groupby('hubble_class')
group_clusters = df_test.groupby('pred_class')


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

fig.suptitle(f'Single dense layer perceptron classification with PCA\n{encoder}')

ax0.set_xlabel('PCA x')
ax0.set_ylabel('PCA y')
ax1.set_xlabel('PCA x')

ax0.set_title('Labels from GalaxyZoo')
for name, group in group_labels:
    ax0.scatter(group.pca_x, group.pca_y, label=name)
ax0.legend()

ax1.set_title('Classes from dense layer perceptron')
for name, group in group_clusters:
    ax1.scatter(group.pca_x, group.pca_y, label=name)
ax1.legend()

fig.savefig(f'results/{encoder}/classifier.pca.png')


# ------ confusion matrix ------
cm = confusion_matrix(
        y_true=y_test.numpy()[:, 0],
        y_pred=y_pred.numpy()
)

labels = ["class_0", "class_1"]
cm_df = pd.DataFrame(
    cm,
    index=pd.Index(labels, name="true"),
    columns=pd.Index(labels, name="pred")
)

result = f'''# Result report

Methodology: 2 perceptrons with 640 features from {encoder}.
Training dataset size: {N_train}
Test dataset size:     {N-N_train}

Confusion matrix:
{cm_df}

Normalized:
{cm_df.div(cm_df.sum(axis=1), axis=0)}
'''

print(result)
with open(f'results/{encoder}/classifier.report.txt', 'w+') as f:
    f.write(result)
