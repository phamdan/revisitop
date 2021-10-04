
import pickle
import os
import numpy as np

from scipy.io import loadmat

from dataset import configdataset
from download import download_datasets
from evaluate import compute_map
import os
data_root = "/content/drive/MyDrive/revisitop/data"
test_dataset = 'oxford5k'
download_datasets(data_root)
# download_features(data_root)

print('>> {}: Evaluating test dataset...'.format(test_dataset)) 
cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))

# load query and database features

print('>> {}: Loading features...'.format(test_dataset))    
with open('vector_query.pickle',"rb") as f:
    Q = pickle.load(f)
with open('vector_dataset.pickle',"rb") as f:
    X = pickle.load(f)
Q=np.array(Q)
Q=Q.T
X=np.array(X)
X=X.T

# perform search
print('>> {}: Retrieval...'.format(test_dataset))
sim = np.dot(X.T, Q)
ranks = np.argsort(-sim, axis=0)

# revisited evaluation
gnd = cfg['gnd']

# evaluate ranks
ks = [1, 5, 10]
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['ok']])
#     g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
    gnd_t.append(g)
mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)
print('>> {}: mAP {:.2f}'.format(test_dataset, np.around(mapE*100, decimals=2)))
