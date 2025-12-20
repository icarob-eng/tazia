import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
import pandas as pd

import tqdm
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

import os



# ------ data treatment ------
ENCODER_NAME = 'zoobot-encoder-convnext_nano'
ENCODER_OUT_FEATURES = 640
TRAIN_FRACTION = 0.8

try:
    os.makedirs(f'results/{ENCODER_NAME}')
except FileExistsError:
    pass

features = pd.read_csv(f'data/{ENCODER_NAME}.feature_vector.csv', sep=' ',
                       header=None, names=['asset_id', *range(ENCODER_OUT_FEATURES)])
features.dropna(axis=0, how='any', inplace=True)
df = pd.read_csv('data/gz2_hart16_classes_simple.csv')

df_features = pd.merge(features, df, on='asset_id', how='inner')  # automatically filters
del df, features

# ------ one-hot encoding and rebalancing ------

hubble_classes = df_features['hubble_class'].unique()
class_columns = [f'hubble_class_{hc}' for hc in hubble_classes]
df_features = pd.get_dummies(df_features, columns=['hubble_class'])  # this drops the actual 'hubble_class' column

class_counts = df_features[class_columns].sum()
print('Class counts before undersampling:', class_counts, '', sep='\n')

resample_size = class_counts.min()  # resample size = size of smallest class
df_features = pd.concat([
    df_features[df_features[col]].sample(resample_size)
    for col in class_columns
])

print('Class counts after undersampling:',
      df_features[class_columns].sum(),
      '',
      sep='\n')

# ------ torch encoding ------
X = data.TensorDataset(torch.tensor(df_features[[*range(ENCODER_OUT_FEATURES)]].to_numpy(), dtype=torch.float32))
y = data.TensorDataset(torch.tensor(df_features[class_columns].to_numpy(), dtype=torch.float32))
del df_features   # save memory ;p

# dividing between train and test
X_train, X_test = torch.utils.data.random_split(X, [TRAIN_FRACTION, 1 - TRAIN_FRACTION])
X_train = X_train[:][0]  # recuperates the tensors from de datasets
X_test = X_test[:][0]
y_train, y_test = torch.utils.data.random_split(y, [TRAIN_FRACTION, 1 - TRAIN_FRACTION])
y_train = y_train[:][0]
y_test = y_test[:][0]
print(f'Total samples: {len(X)}. Train samples: {len(X_train)}. Test samples: {len(X_test)}')
del X, y

# ------ classifier ------
model = nn.Linear(ENCODER_OUT_FEATURES, len(class_counts))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 50
train_loss_history = np.empty(num_epochs)

model.train()
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
plt.savefig(f'results/{ENCODER_NAME}/classifier-loss.png')

model.eval()
with torch.no_grad():
    logits = model(X_test)
    y_pred = torch.sigmoid(logits)


# ------ PCA classes plot ------

pca = PCA(n_components=2)
pca_results = pca.fit_transform(X_test)


df_test = pd.DataFrame({
    'pca_x': pca_results[:, 0],
    'pca_y': pca_results[:, 1],
})
# one-hot to categorical encoding

df_test['hubble_class'] = list(map(lambda i_class: hubble_classes[i_class], np.argmax(y_test, axis=1)))
df_test['pred_class'] = list(map(lambda i_class: hubble_classes[i_class], np.argmax(y_pred, axis=1)))

group_labels = df_test.groupby('hubble_class')
group_classes = df_test.groupby('pred_class')


fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)

fig.suptitle(f'Single dense layer perceptron classification with PCA\n{ENCODER_NAME}')

ax0.set_xlabel('PCA x')
ax0.set_ylabel('PCA y')
ax1.set_xlabel('PCA x')

ax0.set_title('Labels from GalaxyZoo')
for name, group in group_labels:
    ax0.scatter(group.pca_x, group.pca_y, label=name)
ax0.legend()

ax1.set_title('Classes from dense layer perceptron')
for name, group in group_classes:
    ax1.scatter(group.pca_x, group.pca_y, label=name)
ax1.legend()

fig.savefig(f'results/{ENCODER_NAME}/classifier.pca.png')
del group_classes, group_labels

# ------ confusion matrix ------


cm = confusion_matrix(
        y_true=df_test['hubble_class'],
        y_pred=df_test['pred_class']
)

labels = hubble_classes
cm_df = pd.DataFrame(
    cm,
    index=pd.Index(labels, name="true"),
    columns=pd.Index(labels, name="pred")
)

result = f'''# Result report

Methodology: 2 perceptrons with {ENCODER_OUT_FEATURES} features from {ENCODER_NAME}.
Training dataset size: {len(X_train)}
Test dataset size:     {len(X_test)}

Confusion matrix:
{cm_df}

Normalized:
{cm_df.div(cm_df.sum(axis=1), axis=0)}
'''

print(result)
with open(f'results/{ENCODER_NAME}/classifier.report.txt', 'w+') as f:
    f.write(result)
