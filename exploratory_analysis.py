#===============================================================================
# Setup
#===============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

#===============================================================================
# Loading and preparing the data
#===============================================================================

data = pd.read_csv('ionosphere_processed.csv', index_col=0)
data_mat = data.as_matrix()
n_rows, n_cols = data_mat.shape
X = data_mat[:, :n_cols-1]
y = data_mat[:, n_cols-1]

#===============================================================================
# Function definitions
#===============================================================================

def scree_plot(explained_var, save=False):
  plt.figure(figsize=(7, 4))
  plt.bar(left=range(len(explained_var)), height=explained_var)
  plt.title('scree plot')
  plt.xlabel('principal components')
  plt.ylabel('percent variance explained')
  if not save:
    plt.show()
    plt.close('all')
  else:
    plt.savefig('scree_plot.png', bbox_inches='tight')
    plt.close('all')

#===============================================================================
# Summary statistics of the features
#===============================================================================

data_descript = data.describe()
# as can be seen, one of the features equals 
# zero in all observations, so it can be removed
to_remove_cols = data_descript.columns[data_descript.loc['mean', :] == 0]
data_filt = data.loc[
  :, [col for col in data.columns if col not in to_remove_cols]
]
X = data_filt.as_matrix()[:, :data_filt.shape[1]-1]

#===============================================================================
# Covariance and correlation matrices of the features
#===============================================================================

X_centred = X
for i in range(X_centred.shape[1]):
  X_centred[:, i] = X_centred[:, i] - np.mean(X_centred[:, i])

cov_mat = (X.T).dot(X) / (X_centred.shape[0]-1)
diagonals = np.diag(cov_mat).astype(float)
diag_sqrt = np.sqrt(diagonals)
diag_sqrt_inv = 1/diag_sqrt
D_inv = np.zeros((len(diag_sqrt), len(diag_sqrt)))
np.fill_diagonal(D_inv, diag_sqrt_inv)
cor_mat = D_inv.dot(cov_mat).dot(D_inv)

#===============================================================================
# Principal component analysis of the features
#===============================================================================

pca = PCA()
pca.fit(X)
scree_plot(pca.explained_variance_ratio_ * 100)

# determine the number of components required for >= .85 variance explained
# which might filter out some noise and improve the predictions
num_components = 0
cumulative_var = 0
for i in range(len(pca.explained_variance_ratio_)):
  if cumulative_var >= .85:
    break
  else:
    num_components += 1
    cumulative_var += pca.explained_variance_ratio_[i]