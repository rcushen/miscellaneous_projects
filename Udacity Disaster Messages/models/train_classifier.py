import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from joblib import dump

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

def tokenize(tweet):
    '''
    INPUT: a string tweet
    OUTPUT: a normalised, tokenised and lemmatized version of the tweet
    '''
    tokens = word_tokenize(tweet)
    words = [w.lower() for w in tokens if w.isalpha() and w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    stemmed_words = [lemmatizer.lemmatize(w) for w in words ]
    return words

def load_data(database_filepath):
    '''
    INPUT: the filepath of the cleaned tweet database
    OUTPUT: a tuple with X and Y data matrices, as well as the tweet categories
    '''
    engine = create_engine('sqlite:///' + database_filepath, echo=False)
    df = pd.read_sql('SELECT * FROM DisasterResponse', con=engine)
    X = df['message'].values
    Y = df[df.columns[5:]].values
    category_names = list(df.columns[5:])
    return X, Y, category_names

def build_model():
    '''
    OUTPUT: a GridSearCV object, wrapping a pipeline containing:
     - a CountVectorizer that uses the predefined tokenize funcion
     - a TfidfTransformer that transforms the vectorized text
     - a LinearSVC model wrapped in a OneVsRestClassifier
    '''
    pipeline = Pipeline([('cvect', CountVectorizer(tokenizer=tokenize,
                                                    strip_accents='ascii',
                                                    max_features=10000)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', OneVsRestClassifier(LinearSVC(dual=False)))])
    param_grid = {
        'clf__estimator__C': [1, 0.01, 0.001, 0.0001],
        'clf__estimator__tol': [0.001, 0.0001, 0.00001]
    }
    search = GridSearchCV(pipeline, param_grid, iid=False, cv=3, verbose=1)
    return search

def evaluate_model(model, X_test, Y_test):
    preds = model.predict(X_test)
    accuracy = np.mean(preds == Y_test)
    return accuracy

def save_model(model, model_filepath):
    '''
    INPUT: a model/pipeline object and the path to save it to
    '''
    dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating models...')
        test_preds = model.predict(X_test)
        scores = classification_report(Y_test, test_preds)
        print(scores)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
