import sys
import pandas as pd
import numpy as np

from sqlalchemy import create_engine
from joblib import dump

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath, echo=False)
    df = pd.read_sql('SELECT * FROM DisasterResponse', con=engine)
    X = tokenize(df['message'].values)
    Y = df[df.columns[5:]].values
    category_names = df['genre'].values
    return X, Y, category_names

def tokenize(text):
    vectorizer = TfidfVectorizer(
        strip_accents='unicode',
        stop_words='english',
        max_features=10000)
    doc_matrix = vectorizer.fit_transform(text)
    dump(vectorizer, 'data/vectorizer.joblib')
    return doc_matrix.todense()

def build_models():
    models = {
        'naive_bayes': OneVsRestClassifier(MultinomialNB()),
        'logistic_reg': OneVsRestClassifier(LogisticRegression(solver='lbfgs')),
        'linear_svc': OneVsRestClassifier(LinearSVC(dual=False))
    }
    return models

def evaluate_model(model, X_test, Y_test):
    preds = model.predict(X_test)
    accuracy = np.mean(preds == Y_test)
    return accuracy

def save_model(model, model_filepath):
    dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        print('Building models...')
        models = build_models()

        print('Training models...')
        for model in models:
            print('\t...training {}'.format(model))
            models[model].fit(X_train, Y_train)

        print('Evaluating models...')
        results = {
            'naive_bayes': evaluate_model(models['naive_bayes'], X_test, Y_test),
            'logistic_reg': evaluate_model(models['logistic_reg'], X_test, Y_test),
            'linear_svc': evaluate_model(models['linear_svc'], X_test, Y_test)
        }
        for key, val in results.items():
            print('\t {} - {}'.format(key, val))

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        best_model = max(results, key=results.get)
        model = models[best_model]
        save_model(model, model_filepath)
        print('/t...best model: {}'.format(best_model))

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
