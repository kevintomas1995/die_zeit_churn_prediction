import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, fbeta_score
from imblearn.over_sampling import RandomOverSampler
import pickle
import warnings
warnings.filterwarnings('ignore')

from functions_data_handling import data_cleaning, feature_engineering, remove_collinear_features


# churn data original set with 171 columns
df = pd.read_csv("00_data/f_chtr_churn_traintable_nf.csv")

#cleaning data and preparing
df_clean = data_cleaning(df)
df_feat = feature_engineering(df_clean)

# Feature Selection
X = df_feat.drop('churn',axis=1)
Y = df_feat['churn']


# splittin into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("00_data/X_test.csv", index=False)
y_test.to_csv("00_data/y_test.csv", index=False)


numeric_features=['vertragsdauer',
                  'shop_kauf',
                  'nl_aktivitaet',
                  'received_anzahl_6m',
                  'clicked_anzahl_6m',
                  'unsubscribed_anzahl_6m',
                  'openrate_3m',
                  'clickrate_3m',
                  'received_anzahl_bestandskunden_6m',
                  'clicked_anzahl_bestandskunden_6m',
                  'unsubscribed_anzahl_bestandskunden_6m',
                  'received_anzahl_produktnews_6m',
                  'clicked_anzahl_produktnews_6m',
                  'unsubscribed_anzahl_produktnews_6m',
                  'clicked_anzahl_hamburg_6m',
                  'unsubscribed_anzahl_hamburg_6m',
                  'clickrate_hamburg_3m',
                  'received_anzahl_zeitbrief_6m',
                  'clicked_anzahl_zeitbrief_6m',
                  'unsubscribed_anzahl_zeitbrief_6m',
                  'clickrate_zeitbrief_3m']

# scaling:
scaler = StandardScaler()
# if you scale every column, do scaler.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train[numeric_features])
# if you scale every column, do scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test[numeric_features])
# if you scale every column, you can skip this part
X_train_preprocessed = np.concatenate([X_train_scaled, X_train.drop(numeric_features, axis=1)], axis=1)
X_test_preprocessed = np.concatenate([X_test_scaled, X_test.drop(numeric_features, axis=1)], axis=1)


# balancing the data 
ros=RandomOverSampler(random_state=420)
X_train_preprocessed_balanced, y_train_balanced = ros.fit_resample(X_train_preprocessed, y_train)

# voting classifier model from XGBclassifier with different parameter
model_1 = XGBClassifier(scale_pos_weight=2.4700000000000006, eta=0.8, max_depth=16, 
                        sampling_method="gradient_based")

model_2 = XGBClassifier(scale_pos_weight=2.38, eta=0.8, max_depth=11, 
                        sampling_method="gradient_based")

model_3 = XGBClassifier(scale_pos_weight=1.95, eta=0.8, max_depth=16, 
                        sampling_method="gradient_based")

voting_clf_final = VotingClassifier(estimators = [('ROC', model_1),
                                                  ('Rec', model_2),
                                                  ('Prec',model_3)], voting = 'soft')
    
    
# Training the model
voting_clf_final.fit(X_train_preprocessed_balanced, y_train_balanced)

# Predicting on testset 
y_pred_voting_final      = voting_clf_final.predict(X_test_preprocessed)
y_pred_voting_final_prob = voting_clf_final.predict_proba(X_test_preprocessed)

# optimising with new threshold
y_pred_tuned_voting_final = (y_pred_voting_final_prob >= 0.38)[:,1]

# Dispalying the results
print("Recall Score (threshold changed):", round(recall_score(y_test, y_pred_tuned_voting_final)*100, 3),"%")
print("Precision Score (threshold changed):", round(precision_score(y_test, y_pred_tuned_voting_final)*100, 3),"%")
print("ROC Score (threshold changed):", round(roc_auc_score(y_test, y_pred_tuned_voting_final)*100, 3),"%")
print("F-beta Score (threshold changed):", round(fbeta_score(y_test, y_pred_tuned_voting_final, beta=2)*100, 3),"%")