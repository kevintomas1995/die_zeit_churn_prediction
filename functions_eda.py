import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

def add_count(plot):
    '''adds counts to bar and count plots'''
    for p in plot.patches:
        plot.annotate(format(p.get_height(), '.0f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 9), 
                       textcoords = 'offset points')

def value_counts_perc(df, column):
    value_counts = df[column].value_counts()
    
    perc = []
    for i in value_counts:
        perc.append(str(round(i/sum(value_counts)*100,2)))
    
    return perc

def make_crosstab(df, col):
    crosstab = pd.crosstab(df[col],[df["churn"]])
    crosstab.reset_index(level=0, inplace=True)
    crosstab["False_per"]= crosstab[False].div(crosstab[False]+crosstab[True]).multiply(100)
    crosstab["True_per"]= crosstab[True].div(crosstab[False]+crosstab[True]).multiply(100)
    crosstab.drop([False, True], axis=1, inplace=True) 
    return crosstab

def boxplot_1x4 (relevant, df, hue_value=0):
    """
    This function plots several box plots.
    """
    
    relevant = list([relevant])
    
    sns.color_palette('Blues_r')
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(20, 5), sharey=False)
    
    for i in range(0, 4):
        
        sns.boxplot(ax=axes[0], x=df_tot[relevant[0][0]], data=df, orient='h', hue=hue_value)
        sns.boxplot(ax=axes[1], x=df_tot[relevant[0][1]], data=df, orient='h', hue=hue_value)
        sns.boxplot(ax=axes[2], x=df_tot[relevant[0][2]], data=df, orient='h', hue=hue_value)
        sns.boxplot(ax=axes[3], x=df_tot[relevant[0][3]], data=df, orient='h', hue=hue_value)
        
def distplot_1x4 (relevant, df):
    """
    This function plots several box plots.
    """
    
    relevant = list([relevant])
    
    sns.color_palette('Blues_r')
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(20, 5), sharey=False)
    
    for i in range(0, 4):
        
        sns.distplot(ax=axes[0], x=df_tot[relevant[0][0]])
        sns.distplot(ax=axes[1], x=df_tot[relevant[0][1]])
        sns.distplot(ax=axes[2], x=df_tot[relevant[0][2]])
        sns.distplot(ax=axes[3], x=df_tot[relevant[0][3]])
        
def correlation(df):
    """
    This function plots a correlogram.
    """
    #Plot
    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(df.corr())
    ax = sns.heatmap(round(df.corr(), 1),
                     annot=True,
                     mask=mask,
                     cmap="coolwarm",
                     vmax=1,
                     center=0,
                     vmin=-1)
    # Table
    return df.corr().round(2)

# Function to calculate missing values by column
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * mis_val / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(3)
        
        # Print some summary information
        print ("The selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns

# Function for finding descripencies in dataset
def meta(df, transpose=True):
    """
    This function returns a dataframe that lists:
    - column names
    - nulls abs
    - nulls rel
    - dtype
    - duplicates
    - number of diffrent values (nunique)
    """
    metadata = []
    dublicates = sum([])
    for elem in df.columns:

        # Counting null values and percantage
        null = df[elem].isnull().sum()
        rel_null = round(null/df.shape[0]*100, 2)

        # Defining the data type
        dtype = df[elem].dtype

        # Check dublicates
        duplicates = df[elem].duplicated().any()

        # Check number of nunique vales
        nuniques = df[elem].nunique()


        # Creating a Dict that contains all the metadata for the variable
        elem_dict = {
            'varname': elem,
            'nulls': null,
            'percent': rel_null,
            'dtype': dtype,
            'dup': duplicates,
            'nuniques': nuniques
        }
        metadata.append(elem_dict)

    meta = pd.DataFrame(metadata, columns=['varname', 'nulls', 'percent', 'dtype', 'dup', 'nuniques'])
    meta.set_index('varname', inplace=True)
    meta = meta.sort_values(by=['nulls'], ascending=False)
    if transpose:
        return meta.transpose()
    print(f"Shape: {df.shape}")

    return meta

# Correlation diagram
def correlation(df):
    """
    This function plots a correlogram.
    """
    #Plot
    fig, ax = plt.subplots(figsize=(18, 14))
    mask = np.triu(df.corr())
    ax = sns.heatmap(round(df.corr()*100, 0),
                     annot=True,
                     mask=mask, cmap="vlag_r")
    return df.corr().round(2)

# Defining a fucntion for plotting categorical variable and rate of churn
def plot_categorical_variables_bar(data, column_name, rotation = 0, horizontal_adjust = 0, figsize = (15,6), percentage_display = True, plot_defaulter = True,
                                   fontsize_percent = 'xx-small', ha='centre'):
    
    '''
    Function to plot Categorical Variables Bar Plots
    
    Inputs:
        data: DataFrame
            The DataFrame from which to plot
        column_name: str
            Column's name whose distribution is to be plotted
        figsize: tuple, default = (18,6)
            Size of the figure to be plotted
        percentage_display: bool, default = True
            Whether to display the percentages on top of Bars in Bar-Plot
        plot_defaulter: bool
            Whether to plot the Bar Plots for Defaulters or not
        rotation: int, default = 0
            Degree of rotation for x-tick labels
        horizontal_adjust: int, default = 0
            Horizontal adjustment parameter for percentages displayed on the top of Bars of Bar-Plot
        fontsize_percent: str, default = 'xx-small'
            Fontsize for percentage Display
        
    '''
    print(f"Total Number of unique categories of {column_name} = {len(data[column_name].unique())}")
    
    plt.figure(figsize = figsize, tight_layout = True)
    sns.set(style = 'whitegrid', font_scale = 1.2)
    
    #plotting overall distribution of category
    plt.subplot(1,2,1)
    data_to_plot = data[column_name].value_counts().sort_values(ascending = False)
    ax = sns.barplot(x = data_to_plot.index, y = data_to_plot, palette = 'Set1')
    
    if percentage_display:
        total_datapoints = len(data[column_name].dropna())
        for p in ax.patches:
            ax.text(p.get_x() + horizontal_adjust, p.get_height() + 0.005 * total_datapoints, '{:1.02f}%'.format(p.get_height() * 100 / total_datapoints), fontsize = fontsize_percent)
        
    plt.xlabel(column_name, labelpad = 7.5)
    plt.title(f'Distribution of {column_name}', pad = 10)
    plt.xticks(rotation = rotation)
    plt.ylabel('Counts')
    
    #plotting distribution of category for churn
    if plot_defaulter:
        percentage_defaulter_per_category = (data[column_name][data.churn == 1].value_counts() * 100 / data[column_name].value_counts()).dropna().sort_values(ascending = False)

        plt.subplot(1,2,2)
        sns.barplot(x = percentage_defaulter_per_category.index, y = percentage_defaulter_per_category, palette = 'Set2')
        plt.ylabel('Percentage of churner per category')
        plt.xlabel(column_name, labelpad = 7.5)
        plt.xticks(rotation = rotation)
        plt.title(f'Percentage of churner for each category of {column_name}', pad = 10)
    plt.show()
    print()
    print()
    print("*"*100)

# Defining a function for numerical feature with options for different plots
def plot_numerical_variables(data, column_name, plots = ['distplot', 'CDF', 'box', 'violin', 'bar','count'],
                             scale_limits = None, figsize = (12,6), 
                              histogram = True, log_scale = False):
    
    '''
    Function to plot continuous variables distribution
    
    Inputs:
        data: DataFrame
            The DataFrame from which to plot.
        column_name: str
            Column's name whose distribution is to be plotted.
        plots: list, default = ['distplot', 'CDF', box', 'violin']
            List of plots to plot for Continuous Variable.
        scale_limits: tuple (left, right), default = None
            To control the limits of values to be plotted in case of outliers.
        figsize: tuple, default = (20,8)
            Size of the figure to be plotted.
        histogram: bool, default = True
            Whether to plot histogram along with distplot or not.
        log_scale: bool, default = False
            Whether to use log-scale for variables with outlying points.
    '''
    data_to_plot = data.copy()
    if scale_limits:
        #taking only the data within the specified limits
        data_to_plot[column_name] = data[column_name][(data[column_name] > scale_limits[0]) & (data[column_name] < scale_limits[1])]

    number_of_subplots = len(plots)
    plt.figure(figsize = figsize)
    sns.set_style('whitegrid')
    
    for i, ele in enumerate(plots):
        plt.subplot(1, number_of_subplots, i + 1)
        plt.subplots_adjust(wspace=0.25)
        
        if ele == 'CDF':
            #making the percentile DataFrame for both positive and negative Class Labels
            percentile_values_0 = data_to_plot[data_to_plot.churn == 0][[column_name]].dropna().sort_values(by = column_name)
            percentile_values_0['Percentile'] = [ele / (len(percentile_values_0)-1) for ele in range(len(percentile_values_0))]
            
            percentile_values_1 = data_to_plot[data_to_plot.churn == 1][[column_name]].dropna().sort_values(by = column_name)
            percentile_values_1['Percentile'] = [ele / (len(percentile_values_1)-1) for ele in range(len(percentile_values_1))]
            
            plt.plot(percentile_values_0[column_name], percentile_values_0['Percentile'], color = 'darkorange', label = 'Non-churner')
            plt.plot(percentile_values_1[column_name], percentile_values_1['Percentile'], color = 'seagreen', label = 'churner')
            plt.xlabel(column_name)
            plt.ylabel('Probability')
            plt.title('CDF of {}'.format(column_name))
            plt.legend(fontsize = 'medium')
            if log_scale:
                plt.xscale('log')
                plt.xlabel(column_name + ' - (log-scale)')
        
        if ele == 'distplot':  
            sns.distplot(data_to_plot[column_name][data['churn'] == 0].dropna(),
                         label='Non-churner', hist = False, color='darkorange')
            sns.distplot(data_to_plot[column_name][data['churn'] == 1].dropna(),
                         label='churner', hist = False, color='seagreen')
            plt.xlabel(column_name)
            plt.ylabel('Probability Density')
            plt.legend(fontsize='medium')
            plt.title("Dist-Plot of {}".format(column_name))
            if log_scale:
                plt.xscale('log')
                plt.xlabel(f'{column_name} (log scale)')

        if ele == 'violin':  
            sns.violinplot(x='churn', y=column_name, data=data_to_plot, palette="Set2")
            plt.title("Violin-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')
                
        if ele == 'bar':  
            sns.barplot(x='churn', y=column_name, data=data_to_plot, palette="Set2")
            plt.title("Bar-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

        if ele == 'box':  
            sns.boxplot(x='churn', y=column_name, data=data_to_plot, palette="Set2")
            plt.title("Box-Plot of {}".format(column_name))
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')
        
        if ele == 'count':  
            sns.countplot(x=column_name, hue='churn', data=data_to_plot, palette="Set2")
            plt.title("countplot of {}".format(column_name))
            plt.legend(loc='upper right')
            if log_scale:
                plt.yscale('log')
                plt.ylabel(f'{column_name} (log Scale)')

    plt.show()

def dkey(first, second):
    if first == second and second > 0:
        return 1
    else:
         return 0

        
def todumm(cat):
    if cat >1: 
        return 1
    else:
        return 0

        
def create_plz_plot():
    df = pd.read_csv("00_data/f_chtr_churn_traintable_nf.csv")
    df = df.drop(["Unnamed: 0","auftrag_new_id"], axis=1).drop_duplicates()
    df = df.dropna(subset=['ort', 'email_am_kunden'])
    df = df.drop([ "date_x", "avg_churn"], axis=1)


    def crosstab_evaluation(feature_column,target_column,relative=True):
        crosstable = pd.crosstab(feature_column,target_column) 
        if relative:
            crosstable = crosstable.div(crosstable.sum(1),axis=0) 
        return crosstable

    plz1_churn = crosstab_evaluation(df.plz_1,df.churn)
    plz2_churn = crosstab_evaluation(df.plz_2,df.churn)
    plz3_churn = crosstab_evaluation(df.plz_3,df.churn)

    plz_shape_df = gpd.read_file('00_data/plz-gebiete.shp', dtype={'plz': str}) 

    top_cities = {        
        'Berlin': (13.404954, 52.520008),
        'Cologne': (6.953101, 50.935173), 
        'Düsseldorf': (6.782048, 51.227144), 
        'Frankfurt am Main': (8.682127, 50.110924), 
        'Hamburg': (9.993682, 53.551086), 
        'Leipzig': (12.387772, 51.343479), 
        'Munich': (11.576124, 48.137154), 
        'Dortmund': (7.468554, 51.513400), 
        'Stuttgart': (9.181332, 48.777128), 
        'Nuremberg': (11.077438, 49.449820), 
        'Hannover': (9.73322, 52.37052)
    }

    plz_region_df = pd.read_csv( 
        '00_data/zuordnung_plz_ort.csv', sep=',',
        dtype={'plz': str}
    )
    plz_region_df.drop('osm_id', axis=1, inplace=True) 

    # Merge data.
    germany_df = pd.merge( 
        left=plz_shape_df, 
        right=plz_region_df, 
        on='plz', 
        how='inner'
    )
    germany_df.drop(['note'], axis=1, inplace=True)

    def convert_plz_1_to_prob(plz):
        index = str(plz)[0]
        value = plz1_churn.iloc[int(index),1] 
        return value

    def convert_plz_2_to_prob(plz):
        index = str(plz)[0:2]
        value = plz2_churn[plz2_churn.index == index].iloc[0,1] 
        return value

    def convert_plz_3_to_prob(plz):
        index = str(plz)[0:3]
        value = plz3_churn[plz3_churn.index == index].iloc[0,1] 
        return value

    germany_df['churn_plz_1'] = germany_df.plz.apply(lambda x:convert_plz_1_to_prob(x))
    germany_df['churn_plz_2'] = germany_df.plz.apply(lambda x:convert_plz_2_to_prob(x))
    germany_df['churn_plz_3'] = germany_df.plz.apply(lambda x:convert_plz_3_to_prob(x))

    plz_einwohner_df = pd.read_csv( 
        '00_data/plz_einwohner.csv',
        sep=',',
        dtype={'plz': str, 'einwohner': int}
    ) 

    germany_df = pd.merge( 
        left=germany_df, 
        right=plz_einwohner_df, 
        on='plz',
        how='left' 
    )

    fig, ax = plt.subplots(figsize=(16,16))

    germany_df.plot( 
        ax=ax,
        column='churn_plz_3', 
        categorical=False, 
        legend=True, 
        cmap='jet', 
        alpha=0.8,
    )

    for c in top_cities.keys():
        
        ax.text(
            x=top_cities[c][0], 
            y=top_cities[c][1] + 0.08, 
            s=c,
            fontsize=12,
            ha='center',
        )
        
        
        ax.plot( 
            top_cities[c][0], 
            top_cities[c][1], 
            marker='o', 
            c='black', 
            alpha=0.5
        )
        
    ax.set(
        title='Churn rates within germany based on first three digits of postal codes', 
        aspect=1.5,
        facecolor='lightblue'
    ); 

    fig.savefig('00_plots/churn_rate_landscape_plz_3_digit.png',dpi=300)