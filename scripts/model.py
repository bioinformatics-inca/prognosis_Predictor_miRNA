#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pickle

# para evitarmos a exibição dos dados em notacao científica
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[ ]:


from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from numpy import mean, std
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.model_selection import KFold
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.under_sampling import InstanceHardnessThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
np.random.seed(42)

# In[ ]:


#Reload dataset cleaned
dataset = pd.read_csv("top11_clinical.csv", index_col=0)
df = pd.DataFrame(dataset)
print(df.shape)


# In[ ]:


df.head()
print(df.head(4))


# In[ ]:

print(df.progn.value_counts())



# #  Model Development
# 
# ### I define the progn to be predicted (Y)
# 

# In[ ]:


def joinCategories(row):
    if row['progn']== 'Poor'  :
        val = 1
    else:
        val = 0
    return val


# In[ ]:


df['progn'] = df.apply(joinCategories, axis=1)


# In[ ]:


print(df.progn.value_counts())


# In[ ]:


df.head()
print(df.shape)

# In[ ]:


X1=df.drop(['progn'],axis = 1)
y1=df[['progn']] 


# # Resample

# In[ ]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
np.random.seed(42)
# define resampling
smt = RandomOverSampler(random_state=42)
#smt = SMOTE(random_state=42) #deu certo
#smt = SMOTEENN()
# define pipeline
pipeline = Pipeline(steps=[('r', smt)])
# transform the dataset
X, y = pipeline.fit_resample(X1, y1)


# In[ ]:

#save train and test datasets
X.to_csv("X.csv")
y.to_csv("y.csv")
X1.to_csv("X1.csv")
y1.to_csv("y1.csv")
df3 = X.merge(y,left_index=True, right_index=True)
df3.to_csv("df3.csv")


# In[ ]:


print(y.progn.value_counts())


# In[ ]:


print(X.head())


# In[ ]:



np.random.seed(42)



# # LightGBM

# In[ ]:

from imblearn.pipeline import Pipeline, make_pipeline 
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV

sc = StandardScaler()
lgb= lgb.LGBMClassifier(random_state=42, n_jobs=-1)
refcv = RFECV(lgb)
#cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

new_model = Pipeline([('sc', sc), ('Feature Selection',refcv), ('lgb', lgb)])

rs_parameters = {
    'lgb__learning_rate': [0.005,0.01,0.001,0.05],
    'lgb__n_estimators': [20,40,60,80,100],
   'lgb__num_leaves': [6,8,12,16]}


metrics = {'roc_auc','recall', 'f1','accuracy', 'precision'}

gridL = GridSearchCV(new_model,
                         param_grid=rs_parameters,
                         cv=cv_inner,
                         scoring=metrics,
                         refit='roc_auc',
                         return_train_score=False,
                         n_jobs=-1,
                         verbose=True)


gridL.fit(X, y.values.ravel())

grid = open('gridL_grid.pkl', 'wb')
pickle.dump(gridL, grid)
grid.close()


# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# execute the nested cross-validation
scores = cross_val_score(gridL, X, y.values.ravel(), scoring='recall', cv=cv_outer)
# report performance
print('Recall: %.3f (%.3f)' % (mean(scores), std(scores)))

# In[ ]:


print('Best score: %s' % gridL.best_score_)
print('Best params: %s' % gridL.best_params_)

# Resgata as variáveis selecionadas pelo RFECV
indices = gridL.best_estimator_['Feature Selection'].get_support(indices=True)
print(indices)
selected_features = X.iloc[:, indices]

new_model = Pipeline([('sc', sc), ('lgb', lgb)])
gridL = GridSearchCV(new_model,
                         param_grid=rs_parameters,
                         cv=cv_inner,
                         scoring=metrics,
                         refit='roc_auc',
                         return_train_score=False,
                         n_jobs=-1,
                         verbose=True)


gridL.fit(selected_features, y.values.ravel())
# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# execute the nested cross-validation
scores = cross_val_score(gridL, selected_features, y.values.ravel(), scoring='recall', cv=cv_outer)
# report performance
print('Recall: %.3f (%.3f)' % (mean(scores), std(scores)))

# In[ ]:



gbm = gridL.best_estimator_

#save model


grid = open('gridL.pkl', 'wb')
pickle.dump(gridL, grid)
grid.close()

# In[ ]:


np.random.seed(42)
gbm.fit(selected_features,y.values.ravel())


# In[ ]:

from sklearn.metrics import roc_auc_score

score = cross_val_score(gbm,selected_features,y.values.ravel(),cv=cv_outer, scoring ="roc_auc")
print("Mean Score AUC:{:0.3f}".format(score.mean()))

pd.set_option("max_columns",200)
resultsGBM = pd.DataFrame(gridL.cv_results_)
print(resultsGBM.sort_values(by='rank_test_roc_auc')[['params',
                                  'mean_test_recall',
                                  'mean_test_precision',
                                  'mean_test_f1', 'mean_test_roc_auc', 'mean_test_accuracy']][:1])
                                  
                                  

# # Random Forest
# 
# 

# In[ ]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()
rf= RandomForestClassifier(random_state=42)
refcv = RFECV(rf)
#cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, shuffle=True, random_state=42)
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

new_model = Pipeline([('sc', sc), ('Feature Selection',refcv),('rf', rf)])
  
np.random.seed(42)
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 5)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 20, num = 5)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the param grid

grid_params = {'rf__n_estimators': n_estimators,
               'rf__max_features': max_features,
               'rf__max_depth': max_depth,
               'rf__min_samples_split': min_samples_split,
               'rf__min_samples_leaf': min_samples_leaf,
               'rf__bootstrap': bootstrap}


metrics = {'roc_auc','recall', 'f1','accuracy', 'precision'}


grid_RF=GridSearchCV(new_model,
                    param_grid=grid_params,
                    scoring=metrics,
                    verbose=1,
                    refit='roc_auc',
                    cv=cv_inner,
                    n_jobs = -1,
                    return_train_score = True)

grid_RF.fit(X,y.values.ravel())

# configure the cross-validation procedure
cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# execute the nested cross-validation
scores = cross_val_score(grid_RF, X, y.values.ravel(), scoring='recall', cv=cv_outer)
# report performance
print('Recall: %.3f (%.3f)' % (mean(scores), std(scores)))


# In[ ]:


print('Best score: %s' % grid_RF.best_score_)
print('Best params: %s' % grid_RF.best_params_)


# In[ ]:


import pickle
best_rf = grid_RF.best_estimator_

#save model
grid = open('grid_RF.pkl', 'wb')
pickle.dump(grid_RF, grid)
grid.close()


# In[ ]:


np.random.seed(42)
best_rf.fit(X,y.values.ravel())


# In[ ]:


from sklearn.metrics import roc_auc_score

score = cross_val_score(best_rf,X,y.values.ravel(),cv=cv_outer, scoring ="roc_auc")
print("Mean Score AUC:{:0.3f}".format(score.mean()))

pd.set_option("max_columns",200)
resultsRF = pd.DataFrame(grid_RF.cv_results_)
print(resultsRF.sort_values(by='rank_test_roc_auc')[['params',
                                  'mean_test_recall',
                                  'mean_test_precision',
                                  'mean_test_f1', 'mean_test_roc_auc', 'mean_test_accuracy']][:1])
