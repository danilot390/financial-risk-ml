import os
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_df(path):
    """
    Load a dataset from a CSV or Excel File.
    Args:
        path(str) Directory.
    Return:
        (dict) that contains the dataset.
    """
    ext = os.path.splitext(path)[-1].lower()
    if ext =='.csv':
        return pd.read_csv(path)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(path)
    else:    
        print(ext)
        return ValueError('Unsupported file format. Use CSV or Excel.')

def save_df(df, fielname):
    """
    Saves the dataframe as a CSV file in the processed dataset directory
    Args:
        df(dict) Dataset to save.
        path(str) Directory.
    """
    
    path=os.path.join('..', 'data_analysis', 'datasets', 'processed', fielname)
    df.to_csv(path, index=False)
    print(f'Dataset saved successfully at {path}.')

def analizes_df(df):
    """ 
    Summary of the dataframe including shape, head, description, null values, and data types.
    Args:
        df(dict) Dataframe to analysis.
    """
    print("Dataframe Shape: ",df.shape)
    print("\n Head:\n",df.head())
    print("\n Description:\n",df.describe())
    print("\n Total Nulls:\n",df.isnull().sum())
    print("\n Data Types:\n",df.dtypes)

def encode_onehot(df, categorical_columns, numeric_columns):
    """
    Perform one-hot encoding for the specified columns.
    """
    column_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(sparse_output=False), categorical_columns)
        ],
        remainder='passthrough'  
    )

    transformed = column_transformer.fit_transform(df)
    encoded_column_names = column_transformer.named_transformers_['onehot'].get_feature_names_out(categorical_columns)
    return pd.DataFrame(transformed, columns=list(encoded_column_names)+numeric_columns)

def ordinal_encode(df, order, column):
    encoder = OrdinalEncoder(categories=[order])
    df[column] = encoder.fit_transform(df[[column]])
    return df

def box_plot_n(df, columns=None, title='Box Plot of Numeric Columns'):
    """
    Generates box plots for numeric columns to identify outliers.
    Args:
        df(dict) Dataframe to analysis
        columns(array) Contains the columns to include in the graph
        title(str) The title of the plot.
    """
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    plt.figure(figsize=(12, 6))
    df[columns].boxplot(rot=45)  
    plt.title(title)
    plt.show()

def data_cleaning(df):
    """
    Removes NaN values and duplicates records from the dataframe.
    Args:
        df(dict) Dataframe to clean
    Return:
        df(dict) Dataframe cleaned for NaN values & duplicates records.
    """
    try: 
        df = df
        print(f'Data cleaned: Removed NaN Values and {df.duplicated().sum()} duplicates.')
        df=df.dropna().drop_duplicates()
        box_plot_n(df)
    except:
        print('Unexpected Error Found.')
    return df

def winsorize_df(df,columns):
    """ 
    Apllies Winsorization (capping) to limit extreme outliers.
    Args:
        df(dict) Dataframe to capping.
        columns(array) Contains the columns to windorize
    Return:
        df(dict) Dataframe cleaned of outliers.
    """
    for col in columns:
        lower_limit = np.percentile(df[col], 1)
        upper_limit = np.percentile(df[col], 99)  
        df[col] = np.clip(df[col], lower_limit, upper_limit) 
    
    box_plot_n(df, columns=columns, title="Box Plot After Capping Outliers")
    return df

def evaluate_classification(y_true, y_pred, output_dict=True):
    """ 
    Evaluates a classification model using accuracy, AUC, and a classification report heatmap.
    Args:
        y_true(dict) y test values
        y_pred(dict) y predicted values
    """
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}')
    auc = roc_auc_score(y_true, y_pred)
    print(f'ROC AUC: {auc:.4f}')
    print(f'Ginni coefficient: {2*auc-1:.4f}')

    return accuracy, auc

def plot_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict= True)
    df_report = pd.DataFrame(report).drop(['support']).T

    plt.figure(figsize=(8, 5))
    sns.heatmap(df_report, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Classification Report Heatmap')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')
    plt.show()

def plot_lorenz_curve(y_true, y_pred_probs):
    """ 
    Plots the Lorenz curve to visualize  model prediction fairness.
    Args:
        y_true(dict) y test values
        y_pred_probs(dict) y probabilities.
    """

    sorted_indices = np.argsort(y_pred_probs)
    y_true_sorted = np.array(y_true)[sorted_indices]

    cum_total = np.cumsum(y_true_sorted) / np.sum(y_true_sorted)
    cum_population = np.arange(1, len(y_true_sorted)+1)/ len(y_true_sorted)

    plt.figure(figsize=(8, 6))
    plt.plot(cum_population, cum_total, label='Model Curve', color='blue')
    plt.plot([0, 1], [0, 1], label='Baseline (Random Model)', linestyle='--', color='red')
    plt.xlabel('Cumulative Population')
    plt.ylabel('Cumulative Positive Cases')
    plt.title('Lorenz Curve')
    plt.legend()
    plt.grid()
    plt.show()

def preprocessing_def_cred(df, std_scaler=True):
    X = df.drop('default payment next month',axis=1)
    y = df['default payment next month']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if std_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
    return X_train, X_test, y_train, y_test, X, y

def eval_model(y_test, y_pred, y_pred_probs):
    """ 
    Runs a complete evaluation pipeline on a classification model.
    Args:
        y_true(dict) y test values
        y_pred(dict) y predicted values
        y_prob(dict) y probabilities.
    """
    eval = evaluate_classification(y_test, y_pred, y_pred_probs)
    plot_lorenz_curve(y_test, y_pred_probs)

    return eval

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)[:,1]
    print(f'{label}')
    acc, auc = eval_model(y_test, y_pred, y_pred_probs)
    return[label, acc, auc]