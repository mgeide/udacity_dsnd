import numpy as np
import random
import ast
from IPython.display import display_html
import pandas as pd


# categorical_checker()
def categorical_checker(df):
    '''
    Explore dataframe looking for features that may be categorical based on their values (such as having few unique values)
    
    INPUT: dataframe
    
    OUTPUT: dataframe with value info (data type, unique value count, and total values)
    '''
    df_new = pd.DataFrame([df.dtypes,df.nunique(),df.count()]).transpose()
    df_new.columns=["DataType","Unique","Total"]
    df_new["Percent"]=df_new.Unique/df_new.Total
    display_html(df_new[df_new.Percent<.5].transpose())
    return df_new


# random_sampler()
def random_sampler(filename, n_lines, k):
    '''
    Create a random sample from a file
    
    INPUT: 
        filename to pull data from
        number of lines in file
        number of samples to extract
            
    OUTPUT: array containing the sample data
    '''
    sample = []
    
    with open(filename, 'rb') as f:

        random_set = sorted(random.sample(range(n_lines), k), reverse=True)
        lineno = random_set.pop()
        for n, line in enumerate(f):
            if n == lineno:
                sample.append(ast.literal_eval(str(line.rstrip(),"utf-8")))
                if len(random_set) > 0:
                    lineno = random_set.pop()
                else:
                    break
    return sample


# file_len()
def file_len(fname):
    '''
    Return length of file
    
    INPUT: filename
    
    OUTPUT: file length
    '''
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

