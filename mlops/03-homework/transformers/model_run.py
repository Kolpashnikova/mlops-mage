import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(df):
    train_dict = df[['PULocationID', 'DOLocationID']].to_dict(orient='records')

    # Create the DictVectorizer
    vectorizer = DictVectorizer()

    # Fit and transform the data
    X_train = vectorizer.fit_transform(train_dict)

    target = 'duration'
    y_train = df[target].values

    #lr = LinearRegression()
    #lr.fit(X_train, y_train)

    y_train = pd.Series(y_train)

    #return lr.intercept_
    return [X_train, y_train, vectorizer]

