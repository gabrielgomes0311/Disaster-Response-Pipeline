### Table of Contents
1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [Model Stages](#stages)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

Libraries used:

1. pandas
2. numpy
3. sklearn
4. sqlalchemy

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