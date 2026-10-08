# Third-party data and notices

The original Python implementation in this repository does not copy scikit-learn or PyTorch implementation code. It does redistribute an explicitly attributed dataset:

## UCI Wine dataset

Aeberhard, S. & Forina, M. (1992). Wine [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5PC7J

Source: https://archive.ics.uci.edu/dataset/109/wine
License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

The original dataset is based on wine chemical analyses described by Forina et al. in the UCI metadata. Dataset creator names, source, and license are retained for attribution, without implying endorsement.

The exact distributed CSV comes from scikit-learn commit e316dbeeebfd8f38cf293d6443ce81aaa33686d3, sklearn/datasets/data/wine_data.csv. scikit-learn added a metadata header, moved class labels to the last column, and encoded them as 0/1/2 instead of UCI's 1/2/3. No further changes were made to the CSV. Full machine-readable provenance and hashes are in src/pytorch_lab/data/wine_provenance.json.

The upstream scikit-learn BSD-3-Clause notice is retained verbatim in src/pytorch_lab/data/LICENSE.scikit-learn. It is distinct from the UCI dataset license. Dataset provenance is based on UCI's recommended citation rather than inconsistent historical fields in scikit-learn's description file.
