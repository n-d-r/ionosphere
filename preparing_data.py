#===============================================================================
# Setup
#===============================================================================

import requests
import os
import pandas as pd

#===============================================================================
# Downloading the data
#===============================================================================

files_in_folder = os.listdir()
if not 'ionosphere.txt' in files_in_folder:
  url = ('https://archive.ics.uci.edu/ml/machine-learning-databases' +
         '/ionosphere/ionosphere.data')
  response = requests.get(url)
  raw_data = response.text
  with open('ionosphere.txt', 'w') as f:
    f.write(raw_data)
else:
  with open('ionosphere.txt', 'r') as f:
    raw_data = f.read()

#===============================================================================
# Processing data into shape
#===============================================================================

# the .strip() function gets rid of the last \n so that the .split()
# function does not return an empty string as last element
rows = raw_data.strip('\n').split('\n')
rows = [row.split(',') for row in rows]
df = pd.DataFrame(rows)
df.rename(columns={df.columns.values[-1]: 'target'}, inplace=True)
df.to_csv('ionosphere_processed.csv')
df['target'].replace({'g': 1, 'b': 0}, inplace=True)
df.to_csv('ionosphere_processed.csv')