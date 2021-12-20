import numpy as np 
import pandas as pd 

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from category_encoders import BinaryEncoder, HelmertEncoder
from feature_engine.encoding import CountFrequencyEncoder


class cat2num(BaseEstimator, TransformerMixin):
  """cat2num (Categorical to numerical) is a class to deal with categorical variables before being used in ML algorithms.
  Parameters:
  df = None by default. It has to receive a Pandas dataframe;
  categorical_channels_colname = None by default.
      Column name in the df containing the different categorical channels.
      Values could be on a list.
  verbose = False by default.
      Internal parameter for printing;
  """

  def __init__(self, encoder_name=None, cat_cols=None, n_values=None, other_values='other', handle_unknown='ignore'):
    self.encoder_name = encoder_name
    self.cat_cols = cat_cols
    self.n_values = n_values
    self.other_values = other_values
    self.handle_unknown = handle_unknown

    self.encoder_list = {
        'ordinal_encoder':'OE',
        'one_hot_encoder':'OHE',
        'binary_encoder': 'BE',
        'frequency_encoder': 'FE',
        'helmert_encoder': 'HE'
      }

  def choose_encoder(self, encoder_name):

    encoder_name = self.encoder_list[encoder_name]

    if encoder_name == 'OHE':
      enc = OneHotEncoder(handle_unknown=self.handle_unknown, sparse=False)

    elif encoder_name == 'OE':
      enc = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=100000)

    elif encoder_name == 'BE':
      enc = BinaryEncoder()

    elif encoder_name == 'FE':
      enc = CountFrequencyEncoder()

    elif encoder_name == 'HE':
      enc = HelmertEncoder()

    else:
      raise ValueError("The name of the encoder was not found.")
    
    self.encoder = enc

  def select_categorical_columns(self, X):
    self.cat_cols = X.select_dtypes(include=['object', 'category'])

  def fit_nlarge(self, X, y=None):
    self.selected_values_ = {c:X.groupby(c).size().nlargest(self.n_values).index.tolist() for c in self.cat_cols}
    
  def transform_nlarge(self, X):
    X = X.copy()
    for c in self.cat_cols:
      X[c] = X[c].apply(lambda x: x if x in self.selected_values_[c] else self.other_values)
    return X
      
  def normalize_output(self, X):
    if self.encoder_name == 'one_hot_encoder':
      return pd.DataFrame(X, columns=self.encoder.get_feature_names())
    else:
      return X

  def change_encoder(self, encoder_name):
    self.encoder_name = encoder_name

  def fit(self, X, y=None):
    self.choose_encoder(self.encoder_name)

    if self.cat_cols is None:
      self.select_categorical_columns(X)

    if self.n_values is not None:
      self.fit_nlarge(X, y=y)
      return self.encoder.fit(self.transform_nlarge(X[self.cat_cols]))
    else:
      return self.encoder.fit(X[self.cat_cols])

  def transform(self, X):
    if self.n_values is not None:
      X_trans = self.encoder.transform(self.transform_nlarge(X[self.cat_cols]))
    else:
      X_trans = self.encoder.transform(X[self.cat_cols])
    return self.normalize_output(X_trans)
