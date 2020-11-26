'''
Process Data
ETL pipeline that loads, transforms and saves data to database

How to run it:
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
'''

# Import libraries
import sys
import re
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets
    
    Parameters
    ----------
        messages_filepath: Path to the messages file
        categories_filepath: Path to categories file
        
    Rerurns
    -------
        df: Merged dataset of messages and categories
    """    
    
    messages = pd.read_csv(messages_filepath, encoding='latin-1')
    categories = pd.read_csv(categories_filepath, encoding='latin-1')
    
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    """
    Clean dataset
    
    Parameters
    ----------
        df: dataset
        
    Rerurns
    -------
        df: Cleaned dataset
    """     
    
    #Split the values in the categories column on the ; character so that each value becomes a separate column    
    categories = df['categories'].str.split(';',expand=True)

    #Use the first row of categories dataframe to create column names for the categories data
    row = categories[:1]

    #Rename columns of categories with new column names
    category_colnames = []

    for r in row:
        category_colnames.append(row[r][0][:-2])

    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1,  inplace=True)    
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],join='inner', axis=1)
    
    df.drop_duplicates(inplace=True)
    
    return df



def save_data(df, database_filename):
    """
    Save dataset to database
    
    Parameters
    ----------
        df: dataset
        database_filename: db name
        
    Rerurns
    -------
        df: Cleaned dataset
    """   
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DR_table', engine, index=False, if_exists ='replace')


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
