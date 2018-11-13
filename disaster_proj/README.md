# Disaster Response Pipeline Project

### Overview

This project ingests and builds a model from a data set containing real messages that were sent during disaster events. An NLP / machine learning pipeline was created to categorize these events. The project include a web app where an emergency worker can input a new message and get classification results in several categories.

### Instructions
1. Run the following commands in the project's disaster_ide directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the disaster_ide app directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Other Considerations

* The notebooks directory provides the Jupyter notebook scratch work used to build the code within disaster_ide
* There is a known bug (https://github.com/scikit-learn/scikit-learn/issues/10533) within sklearn's joblib that is not allowing me to run GridSearchCV using multiple processing (i.e., n_jobs=-1), so a limited number of parameters are used while n_jobs=1 becuase of the time that it takes to run, once this bug is resolved, change n_jobs=-1 within train_classifier.py build_model() function.
* Future work- build a deep learning model and compare performance against the RandomForst model