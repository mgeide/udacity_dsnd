import numpy as np
import random
import ast
from IPython.display import display_html
import pandas as pd

def skip_rows(samples,length):
    allrows=np.arange(0,length)
    keeprows=np.random.choice(allrows,samples)
    
    #print(allrows,"\n",keeprows)
    
    skiprows=np.delete(allrows, keeprows)
    return (keeprows,skiprows)

def display_side_by_side(args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)
    
def TopCounts(dataframe, colToCount, args, n_rows=20):
    summary_tables=[]
    for colset in args:
        summary_tables.append(dataframe.groupby(colset)[colToCount].count().reset_index(name=f'Count_{colset}').sort_values(f'Count_{colset}',ascending=False).iloc[np.r_[0:n_rows, -n_rows:0]]) 
    display_side_by_side(summary_tables)
    return summary_tables


def categorical_checker(df):
    df_new = pd.DataFrame([df.dtypes,df.nunique(),df.count()]).transpose()
    df_new.columns=["DataType","Unique","Total"]
    df_new["Percent"]=df_new.Unique/df_new.Total
    display_html(df_new[df_new.Percent<.5].transpose())
    return df_new

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def csv_random_sampler(filename, k):
    sample = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()

        random_set = sorted(random.sample(range(filesize), k))

        for i in range(k):
            f.seek(random_set[i])
            # Skip current line (because we might be in the middle of a line) 
            f.readline()
            # Append the next line to the sample set 
            sample.append(str(f.readline().rstrip(),'utf-8').split(','))
    return sample

#Deprecated Dictionary Quick Sampler
#Based on line length

def dict_quick_sampler(filename, k):
    print("Deprecated based upon similarity of speed with more statistically thorough method for large sample sizes.")
    sample = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()

        random_set = sorted(random.sample(range(filesize), k))

        for i in range(k):
            f.seek(random_set[i])
            # Skip current line (because we might be in the middle of a line) 
            f.readline()
            # Append the next line to the sample set 
            sample.append(ast.literal_eval(str(f.readline().rstrip(),"utf-8")))
    return sample


def dict_random_sampler(filename, n_lines, k):
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

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

from IPython.core.display import display,HTML
def shrink_inout():
    return display(HTML('''<style>.jp-InputArea-prompt,.jp-OutputArea-prompt{
        font-size: 6px;
        min-width: 2px;
        margin-left: -80px;
        #display: none; #Use this to get rid of In[] Out[] Alltogether
                 }</style>'''))

def div2html(div,filename='plot.html'):
    tohtml = f'''
    <html>
    <head>
      <script src="https://cdn.plot.ly/plotly-latest.min.js">  </script>
      <style>
        .hovertext text {{
          font-family: Helvetica !important;
          font-size:8px !important;
        }}
      </style>
    </head>
    <body>
      <!-- Output from the Python script above: -->
      {div}
      </body>
    </html>
    '''

    f = open(f'{filename}', 'w')
    f.write(tohtml)
    f.close()

    
def dropzeros(df,axis=0):
    return df.loc[:,(df!=0).any(axis=axis)]

def check_host(host,day,minute):
    day="%02d" % (day,)
    print(f'Checking {host} on Day {day} at Minute {minute}:')
    filename = f'./data/NF_D{day}_30min_CountDTM.csv'
    
    df=pd.read_csv(filename)
    df1=df[(df.id_orig_h==host) & (df["30_minutes"] == minute)]
    df1 = df1.dropna(axis=1)
    print(df1.shape)
    return df1