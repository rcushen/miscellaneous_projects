import sys
import pandas as pd
import re

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    messages.drop_duplicates(subset='id', keep='first', inplace=True)
    categories.drop_duplicates(subset='id', keep='first', inplace=True)
    combined_dataset = messages.merge(categories, how='left', on='id')
    return combined_dataset

def clean_data(df):
    expanded_categories = df['categories'].str.split(pat=';', expand=True)
    colnames = [var.split('-')[0] for var in expanded_categories.iloc[0,:].tolist()]
    expanded_categories.columns = colnames

    for col in expanded_categories.columns:
        expanded_categories[col] = expanded_categories[col].str.extract('-(\d)').astype(int)
        expanded_categories[col] = expanded_categories[col].replace(to_replace=2, value=1)

    df = df.drop(['categories'], axis=1)
    cleaned_dataset = df.merge(expanded_categories, left_index=True, right_index=True)

    def strip_url(string):
        return re.sub('www\\..+\\.', '', string)

    cleaned_dataset['message'] = cleaned_dataset['message'].map(strip_url)
    return cleaned_dataset

def save_data(df, database_filename):
    engine_location = 'sqlite:///' + database_filename
    engine = create_engine(engine_location, echo=False)
    df.to_sql('DisasterResponse', con=engine, if_exists='replace')

def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
