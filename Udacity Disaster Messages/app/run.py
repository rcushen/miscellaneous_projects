import json
import joblib
import plotly
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = stopwords.words('english')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("models/model.joblib")

# calculate some statistics for summary visuals
category_counts = df.iloc[:,5:].values.sum(axis=0).tolist()
categories = list(df.columns[5:])

message_type_counts = df['original'].isnull().value_counts().tolist()
message_type = ['English', 'Translated']

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [{
                'type': 'pie',
                'labels': categories,
                'values': category_counts,
                'textinfo': 'none'
            }
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
            }
        },
        {
            'data': [{
                'type': 'pie',
                'labels': message_type,
                'values': message_type_counts
            }
            ],

            'layout': {
                'title': 'Types of Messages',
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[5:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
