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

# Feature Selection
from boruta import BorutaPy

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
                     continuous_cols, categorical_cols, feature_mask=False):
   
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

                if feature_mask==True:
                    mask = ['x0_2019', 'x1_Midwest', 'x1_NorthEast', 'x1_South', 'x3_0', 'x3_1',
       'x4_1-2x', 'x4_Never', 'x4_Occasionally', 'x4_Regularly Now',
       'x4_Regularly in Past', 'x5_1 Pack', 'x5_1-5 Cigarettes', 'x5_2 Packs',
       'x5_<1 Cigarettes', 'x5_None', 'x6_0', 'x6_1-2X', 'x6_10-19X',
       'x6_20-39X', 'x6_3-5X', 'x6_40+', 'x6_6-9X', 'x7_0', 'x7_1-2X',
       'x7_10-19X', 'x7_20-39X', 'x7_3-5X', 'x7_40+', 'x7_6-9X', 'x8_3-5X',
       'x8_6-9X', 'x8_None', 'x8_Once', 'x8_Twice', 'x9_Female', 'x9_Male',
       'x10_Country', 'x10_Farm', 'x12_No', 'x12_Yes', 'x14_Yes',
       'x15_Graduated HS', 'x16_Graduated HS', 'x17_Yes/Nearly All',
       'x18_Conservative', 'x18_Liberal', 'x18_Very Liberal',
       'x19_Every Week+', 'x19_Never', 'x19_Rarely', 'x20_A Little Important',
       'x20_Not Important', 'x20_Pretty Important', 'x20_Very Important',
       'x21_Above Average', 'x21_Average', 'x22_Average', 'x23_0', 'x24_0',
       'x24_1', 'x25_0', 'x26_0', 'x26_1-2', 'x26_3-5', 'x27_A+',
       "x30_Definitely Won't", 'x31_Definitely Will', 'x37_No', 'x40_0',
       'x41_0', 'x42_1', 'x42_2', 'x42_4-5', 'x42_6-7', 'x43_Never', 'x45_0',
       'x46_0', 'sampling_weight']
                
                    ss = StandardScaler()
                    ohe = OneHotEncoder(sparse=False)

                    cont = continuous_cols
                    cat_cols = categorical_cols

                    scaled = ss.fit_transform(x_t[cont])
                    dummies = ohe.fit_transform(x_t[cat_cols])

                    x_t = pd.concat([pd.DataFrame(scaled, columns=cont), 
                                              pd.DataFrame(dummies, columns=ohe.get_feature_names())], axis=1)
                    x_t = x_t[mask]

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
                    
                    x_val = x_val[mask]
                    
                else:
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

drug_cols = ['weed_hash_lifetime_freq','weed_hash_yr_freq','weed_hash_month_freq','lsd_lifetime_freq','lsd_yr_freq',
             'lsd_month_freq','pysd_lifetime_freq','pysd_yr_freq','pysd_month_freq','coke_lifetime_freq','coke_yr_freq',
             'coke_month_freq','amph_lifetime_freq','amph_yr_freq','amph_month_freq','sedbarb_lifetime_freq','sedbarb_yr_freq',
            'sedbarb_month_freq','tranq_lifetime_freq','tranq_yr_freq','tranq_month_freq','heroin_lifetime_freq','heroin_yr_freq',
             'heroin_month_freq','narcotic_lifetime_freq','narcotic_yr_freq','narcotic_month_freq', 'num_drugs_yr']

