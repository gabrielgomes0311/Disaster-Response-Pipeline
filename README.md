### Table of Contents
1. [Download and Instaltion](#installation)
2. [Project Motivation](#motivation)
3. [Model Stages](#stages)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1. Install Python

2. Install the libraries

. pandas
. numpy
. sklearn
. sqlalchemy
. flask
. plotly

3. Clone the repository

git clone https://github.com/gabrielgomes0311/Disaster-Response-Pipeline.git

4. Set up your database and model.

Run the following commands in the project's root directory.

. To run ETL pipeline that cleans data and stores in database

	'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
	
. To run ML pipeline that trains classifier and saves

	'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'

5. Run the web app

Run the following command in the app's directory to run your web app

	'python run.py'

Go to

	'http://0.0.0.0:3001/'


## Project Motivation <a name="motivation"></a>

The purpose of this project is to build a model that can help in identify disasters. The model receiveis messages and categorizes the messages. 
 
## Model Stages <a name="stages"></a>

1. ETL:

The file process_data.py receives both the messages file and the categories file, with the results of the messages already categorized. Cleans, merges and saves the table in a SQL database.

2. Machine learning pipeline:

The file train_classifier.py receives the messages from the database, uses CountVectorizer, TfidfTransformer, MultiOutputClassifier to train the pipeline. The classification model is the saved in a pickle file.

3. Web app

The file run.py uses plotly to build the graphsd and uses flask to render it in the html template. The app has two pages (go.html and master.html)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Figure Eight for the data.
