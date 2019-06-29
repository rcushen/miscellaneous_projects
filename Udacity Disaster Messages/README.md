# Disaster Response Pipeline Project

## Overview

This project uses a dataset of tweets to train and save a machine learning model that can then be used in a web app to classify tweets in real time. The tweets relate to disaster events, and the categories describe the types of assistance needed by the tweet author.

## Project Structure
The files in this project are structured as follows:
* `data` contains all tweet datafiles and the ETL scripts used to transform them into a form that can be used as input to a machine learning model.
* `models` contains the scripts that train and store the machine learning model.
* `app` contains all files needed to run the web app.

## Instructions
To build the app, follow these instructions:
1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves out
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Modelling Details

The modelling approach taken in this project is split into two steps: feature extraction and classification.
1. **Feature extraction**. Using the 30,000 tweets as input, a bag-of-words representation is constructed, yielding a document-term matrix $D$ in which each row is a tweet and each column is a word. The value in each position of this matrix then signifies the importance of that word relative to the broader corpus of tweets, and is calculated using the tf-idf transformation.
2. **Classification**. Having constructed a numerical representation $D$ of the tweets dataset, a machine learning model can be trained. Three techniques are explored: Naive Bayes, Logistic Regression, and a Linear Support Vector Machine. Since there are 36 possible classes, each of these models must be trained 36 times. The ```OneVsRestClassifier``` is used to this end. The Linear SVC is found to perform best on a held-out test set, and grid search is then used to optimise hyperparameters before the model is saved for use in production.
