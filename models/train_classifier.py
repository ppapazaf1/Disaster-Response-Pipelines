'''
Train Classifier
ML pipeline that trains classifier and saves

How to run it:
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
'''

# import packages

import sys

import re
import numpy as np
import pandas as pd
import sqlite3

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from workspace_utils import *
from workspace_utils import active_session

import pickle

def load_data(database_filepath):
    """
    Load data from Database
    
    Parameter
    ----------
        database_filepath: Path to the database file
        
    Rerurns
    -------
        X: X dimension values
        y: y dimension values
        categ_cols: fearure columns
    """  
    
    # load data from database
    con = sqlite3.connect(database_filepath)

    # Read sqlite query results into a pandas DataFrame
    df = pd.read_sql_query("SELECT * from DR_Table", con)

    con.close()
        
    #Clean Data - Remove feature with all 0 values
    df = df.drop(['child_alone'],axis=1)

    #get fearure columns
    categ_cols = df.columns[5:]
    
    #define X and Y
    X = df.message
    y = df[categ_cols]
    
    return X,y,categ_cols


def tokenize(text):
    """
    Tokenize messages
    
    Parameter
    ----------
        text: test to tokenize
        
    Rerurns
    -------
        clean_tokens: cleaned tokens
    """     
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Build the Model    
        
    Rerurns
    -------
        model: final model
    """    
    # text processing and model pipeline
    model_pipeline = Pipeline([
       ('featr', FeatureUnion([
            ('txt', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))          
         ])),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
        ])
    
    # pipeline.get_params().keys()

    # define parameters for GridSearchCV
    params_grid = {'clf__estimator__n_neighbors':[1,5,7]}
      
    # create gridsearch object and return as final model pipeline
    model = GridSearchCV(model_pipeline, param_grid=params_grid, scoring='f1_micro', n_jobs=-1)


    return model 


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the Model    

    """     
    # predict on test data
    Y_pred = model.predict(X_test)
    
    #evaluate model
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Save the Model    

    """   
    
    # Export model as a pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main process to run all ML pipeline steps
    active_session has been used to avoid sisruption of the process

    """       
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        with active_session():
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