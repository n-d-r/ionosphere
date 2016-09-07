#===============================================================================
# Setup
#===============================================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

#===============================================================================
# Loading and preparing the data
#===============================================================================

data = pd.read_csv('ionosphere_processed.csv', index_col=0)
X = data.as_matrix()[:, :data.shape[1]-1]

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