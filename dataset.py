import pandas as pd
from datetime import datetime
from random import randint


def process_dates(df):

    datecols = ['CreationDate', 'LastAccessDate', 'histref_CreationDate', 'histref_LastAccessDate', 'LastEditDate']
    for col in datecols:
        try:
            df[col] = pd.to_datetime(df[col])
            df[col] = df[col].apply(lambda x: (x - datetime(year=1900, month=1, day=1)).days)
            print(f"converted {col} to integer date")
        except:
            print(f'no success with col {col}')


def match_child_row_numbers(parent, child):
    """ 
    parent child has one to many relationship. Parent rows need to be duplicated to match
    """
    # TODO: add keys as arguments

    # get cardinality
    parent = pd.merge(left=parent, right=child['Id_user'], how="inner", left_on='Id', right_on='Id_user', suffixes=['', '_post'])
    parent['cardinality'] = parent.groupby("Id")["Id"].size()
    # remove users who didn't post
    parent = parent.loc[parent['cardinality'].notna()]
    # repeat based on cardinality
    parent = parent.reindex(parent.index.repeat(parent['cardinality'])) 

    return parent, child


def prepare_data(columns_for_histref:list):
    """ingests data, prepares for table matching

    Args:
        columns_for_histref (list): the columns that can be the matching keys between the tables

    Returns:
        df, df: the two loaded tables
    """    

    print('starting')

    df_post = pd.read_pickle('stats/posts.pickle')
    df_user = pd.read_pickle('stats/user.pickle')

    # drop some columns
    df_post = df_post[['Id', 'OwnerUserId', 'CreationDate', 'LastEditDate']]
    df_user = df_user[['Id', 'CreationDate', 'LastAccessDate', 'AccountId']]

    # remove strange user rows
    df_user = df_user.loc[df_user['Id'] != -1]

    # first join tables by key foreign key (additionally, we'll remove all deleted users this way)
    columns_for_histref = ['CreationDate', 'LastAccessDate']
    df_post = pd.merge(left=df_post, right=df_user[['Id'] + columns_for_histref], how='inner', left_on='OwnerUserId', right_on='Id', suffixes=['', '_user'])
    
    # add histref prefix to histref columns
    for col in columns_for_histref:
        colname = f"{col}_user" if f"{col}_user" in df_post.columns else col
        df_post = df_post.rename(columns={colname: f"histref_{col}"})

    # duplicate user rows to match post rows
    df_user, df_post = match_child_row_numbers(parent=df_user, child=df_post)
    
    # process dates
    process_dates(df_post)
    process_dates(df_user)

    # mutate the post dates to make it synthetic/noisy
    df_post['histref_CreationDate'] = df_post['histref_CreationDate'].apply(lambda x: x + randint(-20,20))
    df_post['histref_LastAccessDate'] = df_post['histref_LastAccessDate'].apply(lambda x: x + randint(-20,20))
    print('done ingesting')

    return df_post, df_user

