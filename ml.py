import numpy as np
import pandas as pd 
from operator import itemgetter

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import warnings
import pickle

warnings.filterwarnings("ignore")








df = pd.read_csv("crop_pred_data.csv")

X = df.drop("label", axis = 1)
y = df["label"]

X = np.array(X)
y = np.array(y)

grid = {"n_estimators": [i for i in range(200, 1200, 10)],
        "max_depth": [i for i in range(1, 30)],
        "max_features": ["auto", "sqrt"],
        "min_samples_split": [i for i in range(1, 6)],
        "min_samples_leaf": [i for i in range(1, 6)]}

clf = RandomForestClassifier(n_jobs=1)


rs_clf = RandomizedSearchCV(estimator=clf,
                            param_distributions=grid, 
                            n_iter=10, 
                            cv=5,
                            verbose=2)

rs_clf.fit(X, y)
params = rs_clf.best_params_

n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth = itemgetter(
     "n_estimators",
     "min_samples_split",
     "min_samples_leaf",
     "max_features",
     "max_depth"
 )(params)

rfc = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                              min_samples_leaf=min_samples_leaf, max_features=max_features,
                              max_depth=max_depth)
rfc.fit(X, y)
pickle.dump(rfc, open('model.pkl', 'wb'))