import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    #Read messages file
    messages = pd.read_csv(messages_filepath)
    
    #Read categories file
    categories = pd.read_csv(categories_filepath)
    
    #Merge files on col=id
    df = pd.merge_ordered(messages,categories,on='id')
    
    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand = True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]
    
    
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    
    #replacing related-2 from related col to related-1
    categories.related.loc[categories.related == 'related-2'] = 'related-1'
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # drop duplicates
    df.drop_duplicates(subset = 'id', inplace = True)
    
    return df

def save_data(df, database_filename):
    try:
        engine = create_engine('sqlite:///{}'.format(database_filename))
        df.to_sql('ResponseMessage', engine, index=False)
        
    except Exception as e:
        print(e)


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