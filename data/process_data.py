import sys
from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath, categories_filepath):  
    """
    load_data: Receives csv and transform on a single dataframe.
    
    inputs:
    . messages_filepath (string) - Messages csv file path
    . categories_filepath (string) - Categories csv file path
    
    outputs: dataframe
    """ 
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = messages.merge(categories, on='id', how='left')
    
    return df


def clean_data(df):
    
    """
    clean_data: Cleaning and transforming the dataframe.
    
    inputs:
    . df (dataframe): Dataframe with messages and categories 
    
    outputs: df
    """
    
    categories = df['categories'].str.split(pat=';',expand=True)
    row = categories.loc[:0]
    category_colnames = [x[:-2] for x in row.values[0].tolist()]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].str[-1]

    categories['related'] = categories['related'].astype('str').str.replace('2', '1')
    
    for column in categories:
        categories[column] = categories[column].astype('int32')

    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df,categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    
    """
    save_data: Saving the dataframe in a sql table.
    
    inputs:
    .df (dataframe) - Modeled dataframe with messages and categories
    .database_filename (string) - Name of SQL database
    
    outputs:
    None
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('df', engine, index=False, if_exists='replace')  
    

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