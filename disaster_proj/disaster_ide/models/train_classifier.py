#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix, classification_report

'''
# Verb extractor logic for additional feature - code from Udacity Nanodegree lecture
# Ignoring this feature for now - cross validation time was limited
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
'''

# Load data function
def load_data(database_filepath):
    '''
    Load data function
    
    INPUT:
        Filepath to database
        
    OUTPUT:
        X - Feature(s) we're going to build a model around (e.g., Message)
        Y - Binary labels tied to feature(s) 
        category_names - Names tied to binary labels
    '''
    try:
        table_name = "disaster"
        engine = create_engine('sqlite:///' + database_filepath)
        df = pd.read_sql_table(table_name, engine)
        X = df.message.values  # Message is the feature
        df.drop(["id", "message", "original", "genre"], axis=1, inplace=True)
        Y = df.values # Binary labels
        category_names = df.columns.values
        return (X, Y, category_names)
    except Exception as e:
        print(e)
        sys.exit("[ERROR] Problem loading data from db.")

# Tokenize data function
def tokenize(text):
    '''
    Tokenize data function
    
    INPUT:
        Text to tokenize
        
    OUTPUT:
        Resulting token list
    '''
    try:
        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens
    except Exception as e:
        print(e)
        sys.exit("[ERROR] Problem tokenizing text.")

# Build model function
def build_model(n_jobs=1):
    '''
    Build model function
    
    INPUT:
        Number of jobs to distribute the GridSearch ("-1" is distribute, "1" is not)
        (Note: there is currently a known bug with sklearn's joblib in doing the distribution)    
            
    OUTPUT:
        Model
    '''
    try:
        pipeline = Pipeline([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
             ])),

            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

        cross_val = 3
        if n_jobs==1:
            # Drastically reduce the params
            parameters = {
                'text_pipeline__vect__ngram_range': [(1, 1)],
                'text_pipeline__vect__max_df': [0.75],
                'text_pipeline__vect__max_features': [None],
                'text_pipeline__tfidf__use_idf': [True],
                'clf__estimator__n_estimators': [100, 200],
                'clf__estimator__min_samples_split': [2, 3],
            }
        else:
            cross_val = 5
                                                     
            # Caution: this will take a long time to GridSearch w/o being able to run multiple jobs
            # recommend n_jobs=-1 after bug fixed in joblib
            parameters = {
                'text_pipeline__vect__ngram_range': [(1, 1), (1, 2)],
                'text_pipeline__vect__max_df': [0.5, 0.75, 1.0],
                'text_pipeline__vect__max_features': [None, 5000, 10000],
                'text_pipeline__tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [50, 100, 200],
                'clf__estimator__min_samples_split': [2, 3, 4],
            }

        cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=n_jobs, cv=cross_val, verbose=100)

        return cv
    except Exception as e:
        print(e)
        sys.exit("[ERROR] Problem building model.")

# Evaluate model
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model function
    
    INPUT:
        - model: model to evaluate
        - X_test: feature values to make a prediction from
        - Y_test: correct classifications for X_test
        - category_names: names of the classification categories
        
    OUTPUT:
        None
    '''
    try:
        Y_pred = model.predict(X_test)
        labels = np.unique(Y_pred)
        
        # note: multiclass-multioutput not supported in confusion_matrix
        #confusion_mat = confusion_matrix(Y_test, Y_pred, labels=labels)
        #accuracy = (Y_pred == Y_test).mean()

        print("Model Evaluation")
        print("================")
        for cat in labels:
            print("Category: ", cat)
            cat_pred = Y_pred[cat].tolist()
            cat_test = Y_test[cat].tolist()
            print(classification_report(cat_test, cat_pred))
        
        #print("\n\nLabels:", labels)
        #print("Confusion Matrix:\n", confusion_mat)
        #print("Accuracy:", accuracy)
        print("\nBest Parameters:", model.best_params_)
    except Exception as e:
        print(e)
        sys.exit("[ERROR] Problem evaluating model.")


# Save model
def save_model(model, model_filepath):
    '''
    Save model function
    
    INPUT:
        None
        
    OUTPUT:
        None
    '''
    try:
        pickle.dump(model, open(model_filepath, 'wb'))
    except Exception as e:
        print(e)
        sys.exit("[ERROR] Problem saving model.")
        
# Main
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        # Note: due to bug in joblib within Anaconda sklearn I'm not able to distribue the GridSearch
        # ref: https://github.com/scikit-learn/scikit-learn/issues/10533
        # n_jobs=1 takes forever, once the bug is resolved- change this to be -1 to distribute
        model = build_model(n_jobs=1)
        
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