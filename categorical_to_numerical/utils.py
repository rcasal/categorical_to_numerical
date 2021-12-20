import numpy as np 
import pandas as pd 

from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, average_precision_score, roc_auc_score, cohen_kappa_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, SMOTENC

from .cat2num import cat2num

def eval_algorithms(X, y,
                    categorical_columns=None, 
                    #categorical_preprocessors={'OHE': OneHotEncoder(handle_unknown="ignore")}, 
                    categorical_preprocessors={'OHE': 'one_hot_encoder'},
                    numerical_preprocessors={'StandarScaler': StandardScaler()},
                    sampling_methods=[],
                    clfs={'Logistic regression': LogisticRegression(max_iter=500)},
                    fillnan = True
                    ):
  # ver si la pasamos el método, o el nombre del método y lo llamamos adentro (así podemos tener le nombre para el ColumnTransformer)

  # remove columns that can cause problems
  X= X.drop(['date','fullVisitorID', 'clientId', 'transactionRevenue'], axis=1, errors='ignore')
  
  # If: returns a list with column names of each type (numerical and categorical)
  if categorical_columns is None:
    numerical_columns_selector = selector(dtype_exclude=['object', 'category'])
    categorical_columns_selector = selector(dtype_include=['object', 'category'])
    numerical_columns = numerical_columns_selector(X) # list
    categorical_columns = categorical_columns_selector(X) # list
  else:
    # cambiar list(df.columns) por el listado de numerical columns (para que no tome las categóricas no elegidas)
    numerical_columns_selector = selector(dtype_exclude=['object', 'category'])
    numerical_columns = numerical_columns_selector(X) # list

  # fill nan (is it necessary an if to decide?)
  if fillnan:
    X[numerical_columns] = X[numerical_columns].fillna(0)  # Warning! Explore, maybe without numerical_columns is ok

  # Instance cat2num class
  encoder = cat2num(cat_cols=categorical_columns, n_values=10)

  # Train Test Split
  X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

  results={}
  sampling_methods.append('noSampl') # without sampling

  for sampling_method in sampling_methods:
    
    if sampling_method != 'noSampl':
      X_train_samp, y_train_samp = sampling(X_train,y_train, categorical_columns, method=sampling_method)
    else:
      X_train_samp, y_train_samp = X_train, y_train

    for clf_name, clf in clfs.items():
      for encoder_name, categorical_preprocessor in categorical_preprocessors.items():
        for num_preproc_name, numerical_preprocessor in numerical_preprocessors.items():
        
          # Model
          encoder.change_encoder(categorical_preprocessor)
          preprocessor = ColumnTransformer([
            ('categorical_preprocessing', encoder, categorical_columns),
            ('numerical_preprocessing', numerical_preprocessor, numerical_columns)])
          
          model = make_pipeline(preprocessor, clf)


          _ = model.fit(X_train, y_train)
          y_prob = model.predict_proba(X_test)[:,1]
          y_prob = y_prob.reshape(-1,1)
          
          # store results
          results_item = metrics(y_prob,y_test)
          method_name = clf_name + '_' + encoder_name + '_' + num_preproc_name + '_' + sampling_method
          results[method_name] = results_item

  # format
  results = pd.DataFrame(results).transpose().reset_index().rename(columns={'index': 'Method Name'})
  return results 


def load_data(df_name='google_analytics_challenge_ds'):

  if df_name=='google_analytics_challenge_ds':
    df = pd.read_csv("/content/drive/MyDrive/Data Science/BADS LAB/Categorical Features Treatment/data.csv", low_memory=False)
    df['transactions'] = np.where(df['transactions']> 0, 1, 0)

  if df_name=='google_analytics_sample':
    df = pd.read_csv("/content/drive/MyDrive/Data Science/BADS LAB/Categorical Features Treatment/ga_sample.csv", low_memory=False)
    df['transactions'] = np.where(df['transactions']> 0, 1, 0)

  return df


def metrics(y_prob,y):

  y_hat = (y_prob>0.5)*1

  # accuracy
  acc = accuracy_score(y, y_hat)

  # balanced accuracy
  bas = balanced_accuracy_score(y, y_hat)

  # F1 score
  f1 = f1_score(y,y_hat)

  # Average precision
  aps = average_precision_score(y, y_prob)

  # ROC AUC
  roc = roc_auc_score(y, y_prob)

  # Cohen Kappa Index
  kappa = cohen_kappa_score(y, y_hat)

  return {'accuracy': acc, 'balanced accuracy': bas, 'f1': f1, 'average precision': aps, 'roc AUC': roc, 'kappa': kappa}


def sampling(X_train,y_train, categorical_columns, method='smote'):

  categorical_columns_index = np.argwhere(X_train.columns.isin(categorical_columns)).ravel()
  
  #print(categorical_columns)
  #print(categorical_columns_index)
  
  if method == 'ros':
    ros = RandomOverSampler(sampling_strategy = 'not majority', random_state=0)
    X_train_samp, y_train_samp = ros.fit_resample(X_train,y_train)
  
  if method == 'smote':
    sm = SMOTENC(categorical_features=categorical_columns_index, sampling_strategy = 'not majority',random_state=0)
    X_train_samp, y_train_samp = sm.fit_resample(X_train, y_train)
   
  return X_train_samp, y_train_samp
