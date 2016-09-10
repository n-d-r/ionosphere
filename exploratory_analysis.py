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

def scree_plot(explained_var, save=True):
  plt.figure(figsize=(7, 4))
  plt.bar(left=range(len(explained_var)), height=explained_var)
  plt.title('scree plot')
  plt.xlabel('principal components')
  plt.ylabel('percent variance explained')
  if not save:
    plt.show()
  else:
    plt.savefig('scree_plot.png', bbox_inches='tight')
  plt.close('all')

def binned_histogram(series, n_bins, min_value=None, max_value=None):
  series = sorted(series)
  if min_value==None:
    min_value = series[0]
  if max_value==None:
    max_value = series[-1]
  if max_value < min_value:
    min_value, max_value = max_value, min_value
  if min_value == max_value:
    return ({(min_value, max_value): len(series)}, max_value / n_bins)

  bin_width = (max_value - min_value) / n_bins
  bin_counts = {(min_value + bin_width*n, 
                 min_value + bin_width*(n + 1)): 0 
                 for n in range(n_bins)}
  bins = sorted(bin_counts.keys())
  bin_index = 0
  for sample_index in range(len(series)):
    for search_index in range(bin_index, n_bins):
      if search_index == (n_bins - 1):
        bin_counts[bins[-1]] += 1
      else:
        if series[sample_index] < bins[search_index][1]:
          bin_counts[bins[search_index]] += 1
          bin_index = search_index
          break

  return (bin_counts, bin_width)

def plot_histogram(bin_counts, bin_width, save=True):
  data_pairs = sorted([(bin[0], count) for bin, count in bin_counts.items()])
  x, y = zip(*data_pairs)

  plt.figure(figsize=(7, 4))
  plt.bar(left=x, height=y, width=bin_width, alpha=.5)
  if save:
    plt.savefig('histogram.png', bbox_inches='tight')
  else:
    plt.show()
  plt.close('all')

def plot_histograms_by_target(save=True, filename=None, **kwargs):
  data_pairs_pos = sorted([(bin[0], count) for bin, count in 
                           kwargs['positive'][0].items()])
  data_pairs_neg = sorted([(bin[0], count) for bin, count in 
                           kwargs['negative'][0].items()])
  x_pos, y_pos = zip(*data_pairs_pos)
  x_neg, y_neg = zip(*data_pairs_neg)

  plt.figure(figsize=(7, 4))
  plt.bar(left=x_pos, height=y_pos, width=kwargs['positive'][1],
          alpha=.5, color='blue')
  plt.bar(left=x_neg, height=y_neg, width=kwargs['negative'][1],
          alpha=.5, color='orange')
  if save:
    plt.savefig('overlapping_histograms_{}.png'.format(filename), 
                bbox_inches='tight')
  else:
    plt.show()
  plt.close('all')

#===============================================================================
# Summary statistics of the features
#===============================================================================

data_descript = data.describe()
# as can be seen, one of the features equals 
# zero in all observations, so it can be removed
to_remove_cols = data_descript.columns[((data_descript.loc['mean', :] == 0) &
                                        (data_descript.loc['std', :] == 0))]
data_filt = data.loc[
  :, [col for col in data.columns if col not in to_remove_cols]
]
X = data_filt.as_matrix()[:, :data_filt.shape[1]-1]

#===============================================================================
# Exploring with histograms
#===============================================================================

plot_histogram(*binned_histogram(series=data['3'], n_bins=100))

g_dataframe = data.loc[data['target']=='g', :]
b_dataframe = data.loc[data['target']=='b', :]
for column in data.columns[:-1]:
  g_series = g_dataframe.loc[:, column]
  b_series = b_dataframe.loc[:, column]
  min_value = min(min(g_series), min(b_series))
  max_value = max(max(g_series), max(b_series))
  plot_histograms_by_target(
    positive=binned_histogram(g_series, 100, min_value, max_value),
    negative=binned_histogram(b_series, 100, min_value, max_value),
    filename=column
  )

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