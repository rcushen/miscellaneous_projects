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
