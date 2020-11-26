# Disaster Response Pipeline Project

https://github.com/ppapazaf1/Disaster-Response-Pipelines

Categorises Disaster Response messages based on content
Includes an ETL pipeline(process_data.py), an ML Pipeline(train_classifier.py), and a Flask based web app (run.py)

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
	
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
	
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app. In order to run it without running the ML pipeline unzip the classifier.7z (7-zip archiver used for high compression ratio.) --> classifier.pkl

    `python run.py`

3. Go to http://0.0.0.0:3001/
