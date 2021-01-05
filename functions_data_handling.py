# importing required library
import pandas as pd


# Function for collinear features finding
def remove_collinear_features(df, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        df: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = df.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    print(drops)
    df = df.drop(columns=drops)
    
    return df


# Defining a function for cleaning the data
def data_cleaning(df):
    
    '''Objective: 
        defining a single function to clean the dataset

    Inputs: 
        df: unsplitted and uncleaned dataframe

    Output: 
        fully cleaned dataframe ready for modelling after feature engineering'''
    
    
    # Drop unwanted columns and duplicate values 
    df = df.drop(["Unnamed: 0","auftrag_new_id"], axis=1).drop_duplicates()

    # Dealing with missing values filling 'kuendigungs_eingangs_datum' with data extraction date
    # Remove the rest of the Nan's from columns 'ort' and 'email_am_kunden'
    df.kuendigungs_eingangs_datum.fillna(value='2020-05-26', inplace=True)
    df = df.dropna(subset=['ort', 'email_am_kunden'])
    
    # Transforming some columns to have only '0' and '1's 
    df['nl_blacklist_sum']  = df.nl_blacklist_sum.apply(lambda x: 1 if x >0 else 0)
    df['nl_bounced_sum']    = df.nl_bounced_sum.apply(lambda x: 1 if x >0 else 0)
    df['nl_sperrliste_sum'] = df.nl_sperrliste_sum.apply(lambda x: 1 if x >0 else 0)
    df['nl_opt_in_sum']     = df.nl_opt_in_sum.apply(lambda x: 1 if x >0 else 0)

    # Renaming the columns to be have consistent name
    df.rename({'openedanzahl_6m':'opened_anzahl_6m',
               'openedanzahl_bestandskunden_6m':'opened_anzahl_bestandskunden_6m',
               'openedanzahl_hamburg_6m': 'opened_anzahl_hamburg_6m',
               'openedanzahl_produktnews_6m': 'opened_anzahl_produktnews_6m',
               'openedanzahl_zeitbrief_6m': 'opened_anzahl_zeitbrief_6m',
               'nl_blacklist_sum': 'nl_blacklist_dum', 
               'nl_bounced_sum': 'nl_bounced_dum', 
               'nl_sperrliste_sum': 'nl_sperrliste_dum',
               'nl_opt_in_sum': 'nl_opt_in_dum'
               }, axis=1, inplace=True)

    # Dropping outlier 
    df.drop(index=df[df["shop_kauf"] >= 100].index, inplace=True)
    df.drop(index=df[df["cnt_abo"] >= 21].index, inplace=True)
    df.drop(index=df[df["received_anzahl_bestandskunden_1w"] >= 4].index, inplace=True)
    df.drop(index=df[df["received_anzahl_produktnews_6m"] >= 14].index, inplace=True)
    df.drop(index=df[df["received_anzahl_hamburg_6m"] >= 130].index, inplace=True)
    df.drop(index=df[df["received_anzahl_zeitbrief_1w"] >= 3].index, inplace=True)
    df.drop(index=df[df["received_anzahl_zeitbrief_1m"] >= 7].index, inplace=True)
    df.drop(index=df[df["received_anzahl_zeitbrief_3m"] >= 16].index, inplace=True)
    df.drop(index=df[df["received_anzahl_zeitbrief_6m"] >= 29].index, inplace=True)
    df.drop(index=df[df["opened_anzahl_zeitbrief_1w"] >= 3].index, inplace=True)
    df.drop(index=df[df["opened_anzahl_zeitbrief_1m"] >= 7].index, inplace=True)
    df.drop(index=df[df["opened_anzahl_zeitbrief_3m"] >= 16].index, inplace=True)
    df.drop(index=df[df["opened_anzahl_zeitbrief_6m"] >= 29].index, inplace=True)

    # Dropping open and clickrates anamolies
    df.drop(index=df[df["openrate_bestandskunden_3m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_bestandskunden_3m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_bestandskunden_1w"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_bestandskunden_1w"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_bestandskunden_1m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_bestandskunden_1m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_produktnews_3m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_produktnews_3m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_produktnews_1w"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_produktnews_1w"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_produktnews_1m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_produktnews_1m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_hamburg_3m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_hamburg_3m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_hamburg_1w"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_hamburg_1w"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_hamburg_1m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_hamburg_1m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_zeitbrief_3m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_zeitbrief_3m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_zeitbrief_1w"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_zeitbrief_1w"] > 1.0].index, inplace=True)
    df.drop(index=df[df["openrate_zeitbrief_1m"] > 1.0].index, inplace=True)
    df.drop(index=df[df["clickrate_zeitbrief_1m"] > 1.0].index, inplace=True)

    # Dropping corelated feature columns
    df.drop(['avg_churn', 'ort', 'date_x', 'training_set', 'zon_rawr','zon_community',
             'zon_app_sonstige','zon_schach','zon_blog_kommentare','zon_quiz','zon_boa',
             'zon_kommentar','plz_1','plz_2','cnt_abo_diezeit','cnt_abo_diezeit_digital',
             'cnt_abo_magazin','received_anzahl_1w','received_anzahl_1m','received_anzahl_3m',
             'opened_anzahl_1w','opened_anzahl_1m','opened_anzahl_3m','clicked_anzahl_1w',
             'clicked_anzahl_1m','clicked_anzahl_3m','unsubscribed_anzahl_1w',
             'unsubscribed_anzahl_1m','unsubscribed_anzahl_3m','openrate_1w',
             'clickrate_1w','openrate_1m','clickrate_1m','received_anzahl_bestandskunden_1w',
             'received_anzahl_bestandskunden_1m','received_anzahl_bestandskunden_3m',
             'opened_anzahl_bestandskunden_1w','opened_anzahl_bestandskunden_1m',
             'opened_anzahl_bestandskunden_3m','clicked_anzahl_bestandskunden_1w',
             'clicked_anzahl_bestandskunden_1m','clicked_anzahl_bestandskunden_3m',
             'unsubscribed_anzahl_bestandskunden_1w','unsubscribed_anzahl_bestandskunden_1m',
             'unsubscribed_anzahl_bestandskunden_3m','openrate_bestandskunden_1w',
             'clickrate_bestandskunden_1w','openrate_bestandskunden_1m','clickrate_bestandskunden_1m',
             'received_anzahl_produktnews_1w','received_anzahl_produktnews_1m',
             'received_anzahl_produktnews_3m','opened_anzahl_produktnews_1w',
             'opened_anzahl_produktnews_1m','opened_anzahl_produktnews_3m',
             'clicked_anzahl_produktnews_1w','clicked_anzahl_produktnews_1m',
             'clicked_anzahl_produktnews_3m','unsubscribed_anzahl_produktnews_1w',
             'unsubscribed_anzahl_produktnews_1m','unsubscribed_anzahl_produktnews_3m',
             'openrate_produktnews_1w','clickrate_produktnews_1w','openrate_produktnews_1m',
             'clickrate_produktnews_1m','received_anzahl_hamburg_1w','received_anzahl_hamburg_1m',
             'received_anzahl_hamburg_3m','opened_anzahl_hamburg_1w','opened_anzahl_hamburg_1m',
             'opened_anzahl_hamburg_3m','clicked_anzahl_hamburg_1w','clicked_anzahl_hamburg_1m',
             'clicked_anzahl_hamburg_3m','unsubscribed_anzahl_hamburg_1w',
             'unsubscribed_anzahl_hamburg_1m','unsubscribed_anzahl_hamburg_3m',
             'openrate_hamburg_1w','clickrate_hamburg_1w','openrate_hamburg_1m',
             'clickrate_hamburg_1m','received_anzahl_zeitbrief_1w','received_anzahl_zeitbrief_1m',
             'received_anzahl_zeitbrief_3m','opened_anzahl_zeitbrief_1w','opened_anzahl_zeitbrief_1m',
             'opened_anzahl_zeitbrief_3m','clicked_anzahl_zeitbrief_1w','clicked_anzahl_zeitbrief_1m',
             'clicked_anzahl_zeitbrief_3m','unsubscribed_anzahl_zeitbrief_1w',
             'unsubscribed_anzahl_zeitbrief_1m','unsubscribed_anzahl_zeitbrief_3m',
             'openrate_zeitbrief_1w','clickrate_zeitbrief_1w','openrate_zeitbrief_1m',
             'clickrate_zeitbrief_1m',], axis=1, inplace=True)

    return df 


def feature_engineering(df):
    
    '''Objective: 
        defining a single function to incorporate feature engineering

    Inputs: 
        df: cleaned data set from output of the function "data_cleaning()" : 

    Output: 
        fully cleaned and feature engineered dataframe ready for modelling'''
    
    
    # transform following column into only two values
    relevant = ["zon_zp_grey", "zon_premium", "zon_sonstige", "zon_zp_red"]
    for i in relevant:
        df[i] = df[i].apply(lambda x: 0 if x==0 or x==1 else 1)

    # New column created for abonnement with only two values
    df["more_than_one_cnt_abo"] = df["cnt_abo"].apply(lambda x: 0 if x==0 else 1)

    # creating additional 'umwandlungsstatus2' columns
    def dkey(first, second):
        if first == second and second > 0:
            return 1
        else:
            return 0
    df["cnt_dkey_equals_cnt_abo"]     = df.apply(lambda x: dkey(x["cnt_umwandlungsstatus2_dkey"], 
                                                                x["cnt_abo"]), axis=1)
    df["cnt_dkey_more_than_one"]      = df.cnt_umwandlungsstatus2_dkey.apply(lambda x: 1 if x >0 else 0)

    # Converting datelike feature to Datatype 'datetime' for future use
    df["kuendigungs_eingangs_datum"]  = pd.to_datetime(df.kuendigungs_eingangs_datum, format="%Y-%m-%d")
    df["liefer_beginn_evt"]           = pd.to_datetime(df.liefer_beginn_evt, format="%Y-%m-%d")

    # Creating new columns by tranforming the existing ones
    df["vertragsdauer"]               = df["kuendigungs_eingangs_datum"] - df["liefer_beginn_evt"]
    df["vertragsdauer"]               = df["vertragsdauer"].apply(lambda x: x.days)
    df["abo_registrierung_min"]       = df.abo_registrierung_min.apply(lambda x: x.split()[0])
    df["abo_registrierung_min_year"]  = df.abo_registrierung_min.apply(lambda x: x.split("-")[0])
    df["abo_registrierung_min_month"] = df.abo_registrierung_min.apply(lambda x: x.split("-")[1])
    df["nl_registrierung_min"]        = df.nl_registrierung_min.apply(lambda x: x.split()[0])
    df["nl_registrierung_min_year"]   = df.nl_registrierung_min.apply(lambda x: x.split("-")[0])
    df["nl_registrierung_min_month"]  = df.nl_registrierung_min.apply(lambda x: x.split("-")[1])

    # Dropping original feature columns after creation of new columns
    df = df.drop(["abo_registrierung_min", "nl_registrierung_min",
                  "kuendigungs_eingangs_datum", "liefer_beginn_evt", 
                  "lesedauer","cnt_abo", "cnt_umwandlungsstatus2_dkey"], axis=1)

    # Dropping highly corelated features to build more general predictive model
    df = remove_collinear_features(df, 0.7)

    # Creation of dummy feature columns
    other = ["zon_che_opt_in", "zon_sit_opt_in","zon_zp_grey","zon_premium","zon_sonstige",
             "zon_zp_red","nl_zeitbrief","nl_zeitshop","nl_zeitverlag_hamburg","nl_fdz_organisch"]
    
    categoric_features = list(df.columns[df.dtypes==object]) + other
    df = pd.get_dummies(df, columns=categoric_features, drop_first=True)
    
    # Return the final dataframe
    return df


    

