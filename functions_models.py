# Basic packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Library for timing
from time import time

# Libraries for model evaluation
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Ignore warnings while plotting
import warnings
warnings.filterwarnings("ignore")

'==============================================================================================================================='
'==============================================================================================================================='

# Function to convert value range of a column
def todumm(cat):
    '''
    Objective:
        Function to compress a value range (0, 1, 2) to a binary range (0, 1).
    '''
    if cat >1: 
        return 1
    else:
        return 0

'==============================================================================================================================='
'==============================================================================================================================='

# Function to run pre-defined models
def run_models(X_train, X_test, y_train, y_test, models):
    '''
    Objective:
        Running several (predefined) ML algorithms with one function and capture the results.
    Input:
        1) First the models have to be instantiated with its classifier as well as the used parameter, e.g.:
               /// model_XGB = XGBClassifier(n_estimators = 200, gamma = 100, 
                                            learning_rate = 0.01, max_depth = 12, booster = 'gbtree',
                                            scale_pos_weight = 1.5, objective='binary:logistic') ///
        2) Create list with instantiated models (model names)
    Output:
        1) results_list -->> List containing the results by algorithm.
        2) predicted_target_values_list -->> List containing the predicted target values.
    '''
    # Lists to capturing the results
    results_list = []
    predicted_target_values_list = []
    # Looping trough the models
    for model in models:
        results, predicted_target_values = predict(X_train, X_test, y_train, y_test, model)
        results_list.append(results)
        predicted_target_values_list.append(predicted_target_values)
    return results_list, predicted_target_values_list

'==============================================================================================================================='
'==============================================================================================================================='

# Defining a function for prediction
def predict(X_train, X_test, y_train, y_test, model):
    '''
    Objective:
        Make predictions with a pre-defined set of ML algorithms.
    Inputs:
        - X_train:  Features training set
        - X_test:   Features testing set
        - y_train:  Income training set
        - y_test:   Income testing set
        - Model:    The ML algorithm (model) used to train and predict.
    Output:
        - results -->> as dictionary
        - predicted_target_values -->> as dictionary
    '''

    # Dictionary capturing the results by model
    results = {}
    # Dictionary capturing the target values by model
    predicted_target_values = {}

    # Train the model by fitting the train data set (X_train y_train)
    start = time() # Get start time
    model = model.fit(X_train ,y_train)
    end = time() # Get end time

    # Add the model name as key
    model_name = model.__class__.__name__
    results['model_name'] = model_name

    # Calculate the training time
    results['train_time'] = end - start

    # Get the predictions on the test and train set
    start = time() # Get start time
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    end = time() # Get end time

    '''
    Capturing predicted target values (y_test_pred and y_train_pred)
    '''
    # Add the model name as key
    predicted_target_values['model_name'] = model_name
    # Capture y_train
    predicted_target_values['y_train'] = y_train.values.tolist()
    # Capture y_test
    predicted_target_values['y_test'] = y_test.values.tolist()
    # Capture y_train_pred
    predicted_target_values['y_pred_train'] = y_pred_train
    # Capture y_test_pred
    predicted_target_values['y_pred_test'] = y_pred_test

    '''
    Evaluation by different parameters
    '''
    # Calculate the total prediction time
    results['pred_time'] = end - start
    # Compute accuracy on the train set
    results['acc_train'] = accuracy_score(y_train, y_pred_train)
    # Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, y_pred_test)
    # Compute Precision_score on the train set
    results['precision_train'] = precision_score(y_train, y_pred_train)
    # Compute Precision_score on the test set
    results['precision_test'] = precision_score(y_test, y_pred_test)
    # Compute Recall_score on the train set
    results['recall_train'] = recall_score(y_train, y_pred_train)
    # Compute Recall_score on the test set
    results['recall_test'] = recall_score(y_test, y_pred_test)

    # Final results
    print('Finished working out in the gym: {} '.format(model.__class__.__name__))
    # Return the results
    return results, predicted_target_values

'==============================================================================================================================='
'==============================================================================================================================='

# Function to output the report on ML models run
def ml_reporting(results_list, predicted_target_values_list):
    '''
    Objective:
        Make predictions with a pre-defined set of ML algorithms.
    Inputs:
        - X_train:  Features training set
        - X_test:   Features testing set
        - y_train:  Income training set
        - y_test:   Income testing set
        - Model:    The ML algorithm (model) used to train and predict.
    Output:
        - results -->> as dictionary
        - predicted_target_values -->> as dictionary
    '''

    df_res = pd.DataFrame(results_list)
    df_pred = pd.DataFrame(predicted_target_values_list)

    for i, results in enumerate(results_list):
        ### Header ###
        print (('\033[1m \033[4m' + 'Reporting for {0}:\n' + '\033[0m').format(results['model_name']))

        ### Individual report ###
        display(df_res.loc[[i]])

        ### Classification report ###
        print(('\033[1m' + '\nClassification Report for {0} :\n\n' + '\033[0m').format(results['model_name']), classification_report(df_pred.iloc[i]['y_test'], df_pred.iloc[i]['y_pred_test']))
        
        ### Plot confusion matrix as heatmap ###
        # Generate confusion matrix for individual model
        conf_matrix = confusion_matrix(df_pred.iloc[i]['y_test'], df_pred.iloc[i]['y_pred_test'])
        # Set up labels
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()/ np.sum(conf_matrix)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        # Set up plot layout
        plt.figure(figsize = (4, 4))
        sns.set(font_scale=1.15) #for label size (x- & y-labels)
        # Set up plot (annot_kws for setting up the font size of numbers)
        sns.heatmap(conf_matrix, cmap='YlGnBu', annot=labels, annot_kws={'size': 13}, cbar=False, fmt='')
        # Set up title and y- and x-label
        plt.title(('Confusion Matrix - {0}\n').format(results['model_name']), fontsize=(14), fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        # Shop plot to make sure of right ordered report
        plt.show()
        print('\n' + '='*100)
        print('='*100 + '\n')


'==============================================================================================================================='
'==============================================================================================================================='

# Function to export the confusion matrix as .png
def ml_reporting_exp_figs(results_list, predicted_target_values_list, path):
    '''
    Objective:
        This function plots the resulting confusion matrix and exports it as .png-file
    Inputs:
        - results_list
        - predicted_target_values_list
    Output:
        - Confusion matrix as .png
    '''
    df_res = pd.DataFrame(results_list)
    df_pred = pd.DataFrame(predicted_target_values_list)
    
    for i, results in enumerate(results_list):     
        ### Plot confusion matrix as heatmap ###
        # Generate confusion matrix for individual model
        conf_matrix = confusion_matrix(df_pred.iloc[i]['y_test'], df_pred.iloc[i]['y_pred_test'])
        # Set up labels
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ['{0:0.0f}'.format(value) for value in conf_matrix.flatten()]
        group_percentages = ['{0:.2%}'.format(value) for value in conf_matrix.flatten()/ np.sum(conf_matrix)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        # Set up plot layout
        plt.figure(figsize = (4, 4), frameon=False)
        sns.set(font_scale=1.15) #for label size (x- & y-labels)
        # Set up plot (annot_kws for setting up the font size of numbers)
        sns.heatmap(conf_matrix, cmap='YlGnBu', annot=labels, annot_kws={'size': 13}, cbar=False, fmt='')
        # Set up title and y- and x-label
        plt.title(('Confusion Matrix - {0}\n').format(results['model_name']), fontsize=(14), fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        # Safe figure as .png
        plt.savefig((path + 'confusion_matrix_{0}.png').format(results['model_name']), transparent=True, bbox_inches='tight', dpi=300)
        # Handling plot apperance
        plt.close()
        # Status
        print (('{0} exported.').format(results['model_name']))

'==============================================================================================================================='
'==============================================================================================================================='

# Function to export the results into an excel-file
def ml_reporting_exp_tables(results_list, predicted_target_values_list, path):
    '''
    Objective:
        Exporting the results as .xlsx-file
    Inputs:
        - results_list
        - predicted_target_values_list
    Output:
        - Classification reports as .xlsx
    '''
    # Create data frames out of the results and predictions
    df_res = pd.DataFrame(results_list)
    df_pred = pd.DataFrame(predicted_target_values_list)

    # Export results from custom report
    with pd.ExcelWriter(path + 'baseline_output_custom.xlsx') as writer:
        df_res.to_excel(writer, sheet_name='individual_results')
        df_pred.to_excel(writer, sheet_name='individual_predictions')

    # Export results from confusion matrices
    # Set up list for data frames
    #df_cm_list = []

    # Loop to write results into a list
    #for i, results in enumerate(results_list):
        # Generate confusion matrix as data frame
    #    df_cm = pd.DataFrame(classification_report(df_pred.iloc[i]['y_test'], df_pred.iloc[i]['y_pred_test'], output_dict=True)).transpose()
    #    df_cm_list.append(df_cm)
        # Status
    #    print (('{0} exported.').format(results['model_name']))

    #writer = pd.ExcelWriter('00_plots/baseline_output_confusion_matrices.xlsx')
    #for i, cm in enumerate(df_cm_list):
    #    cm.to_excel(writer, sheet_name=f'confusion_matrices_{i}')

'==============================================================================================================================='
'==============================================================================================================================='