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
from sklearn.naive_bayes import GaussianNB

def load_data(database_filepath):
    engine_location = 'sqlite:///' + database_filepath
    engine = create_engine(engine_location, echo=False)
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

def build_model():
    return OneVsRestClassifier(GaussianNB())

def evaluate_model(model, X_test, Y_test, category_names):
    preds = model.predict(X_test)
    accuracy = np.mean(preds == Y_test)
    print(accuracy)

def save_model(model, model_filepath):
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

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
