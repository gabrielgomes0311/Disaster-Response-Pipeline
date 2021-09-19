# import libraries

import sys

import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
import re

import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

from nltk.corpus import stopwords

from sklearn import metrics

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    
    """
    load_data: Loading data from SQL and preparing data to be modeled.
    
    parameters:
    . database_filepath (string) - Path of SQL table
    
    return:
    . X
    . Y
    . category_names
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('df', engine)
    X = df.message.values
    Y = df.iloc[:, 4:].values
    category_names = np.array(df.drop(columns=['id', 'message', 'original', 'genre']).columns)
    
    return X,Y,category_names


def tokenize(text):
    
    """
    tokenize: Tokenizing text to be modeled.
    
    parameters:
    . text (string) - Content for modelling
    
    return: tokens
    """
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    tokens = word_tokenize(text)
    
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_model():

    """
    build_model: NLP model
    
    inputs: None
    
    outputs: model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10]
    }

    model = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=None, verbose=12, n_jobs=-1)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    evaluate_model: Test result of NLP model
    
    inputs:
    . model - NLP pipeline
    . X_test (array) - Independent features for testing
    . Y_test (array) - Dependent features for testing
    
    outputs: df_test
    """
       
    y_pred = model.predict(X_test)
    y_test_arr = np.array(Y_test)

    df_test = pd.DataFrame()
    for i,names in enumerate(category_names):
        accuracy = accuracy_score(y_test_arr[:, i], y_pred[:, i])
        f1 = f1_score(y_test_arr[:, i], y_pred[:, i], average='micro')
        precision = precision_score(y_test_arr[:, i], y_pred[:, i], average='micro')
        recall = recall_score(y_test_arr[:, i], y_pred[:, i], average='micro')
        df_test = df_test.append({'Category Names':names,'Accuracy':accuracy,'F1 Score':f1,'Precision':precision,'Recall':recall},ignore_index = True)
    
    return df_test

def save_model(model, model_filepath):

    """
    save_model: Save model in a pickle file
    
    inputs:
    . model - NLP pipeline
    . model_filepath (string) - Model file path
    
    outputs: None
    """
    
    model_pickle = open(model_filepath,'wb')

    pickle.dump(model, model_pickle)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        #database_filepath, model_filepath = ('DisasterResponse.db', 'model.pickle')
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