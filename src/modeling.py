'''modeling: a package containing classes/functions that iterate through different permutations of classification models, and logs/graphs its metrics.'''

# general imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

# Modeling
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from bayes_opt import BayesianOptimization
import lightgbm
import catboost

# Evaluation
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

class ModelHistory():
    '''Keeps track of performance results for various models'''
    
    
    def __init__(self, X_train, y_train, X_test, y_test, history={}):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.history = history
        
    
    def kfold_validation(self, classifier, name,
                     continuous_cols, categorical_cols):
   
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

        val_recall = []
        val_prec = []
        val_acc = []
        roc_auc = []

        for train_ind, val_ind in skf.split(self.X_train, self.y_train):
            x_t = self.X_train.iloc[train_ind]
            y_t = self.y_train.iloc[train_ind]


            if len(continuous_cols)==0:

                ohe = OneHotEncoder(sparse=False)

                cat_cols = categorical_cols

                dummies = ohe.fit_transform(x_t[cat_cols])

                x_t = pd.DataFrame(dummies, columns=ohe.get_feature_names())

                x_val = self.X_train.iloc[val_ind]
                y_val = self.y_train.iloc[val_ind]

                dums = ohe.transform(x_val[cat_cols])

                x_val = pd.DataFrame(dums, columns=ohe.get_feature_names())

            elif len(continuous_cols)!=0:

                ss = StandardScaler()
                ohe = OneHotEncoder(sparse=False)

                cont = continuous_cols
                cat_cols = categorical_cols

                scaled = ss.fit_transform(x_t[cont])
                dummies = ohe.fit_transform(x_t[cat_cols])

                x_t = pd.concat([pd.DataFrame(scaled, columns=cont), 
                                          pd.DataFrame(dummies, columns=ohe.get_feature_names())], axis=1)

                x_val = self.X_train.iloc[val_ind]
                y_val = self.y_train.iloc[val_ind]


                sc = ss.transform(x_val[cont])
                dums = ohe.transform(x_val[cat_cols])

                x_val = pd.concat([pd.DataFrame(sc, columns=cont), 
                                      pd.DataFrame(dums, columns=ohe.get_feature_names())], axis=1)


                x_val = self.X_train.iloc[val_ind]
                y_val = self.y_train.iloc[val_ind]


                sc = ss.transform(x_val[cont])
                dums = ohe.transform(x_val[cat_cols])

                x_val = pd.concat([pd.DataFrame(sc, columns=cont), 
                                          pd.DataFrame(dums, columns=ohe.get_feature_names())], axis=1)

            clf = classifier

            clf.fit(x_t, y_t)

            val_recall.append(recall_score(y_val, clf.predict(x_val)))
            val_prec.append(precision_score(y_val, clf.predict(x_val)))
            val_acc.append(accuracy_score(y_val, clf.predict(x_val)))
            roc_auc.append(roc_auc_score(y_val, clf.predict(x_val)))
        
        self.history[name] = {'recall': val_recall, 'precision': val_prec, 'accuracy': val_acc, 'roc_auc':roc_auc}


        return val_recall, val_prec, val_acc, roc_auc, clf


