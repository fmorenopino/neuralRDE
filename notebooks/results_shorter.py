# %%
import sys; sys.path.append('../')
sys.path.append('/nfs/home/fernandom/github/neuralRDE')
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

# %%
from experiments.parse_results import parse_results, parse_results_bidmc

# %%
def fix_index(all_results):
    all_results.reset_index(inplace=True)
    all_results['depth'] = all_results['depth'].astype(int)
    all_results['step'] = all_results['step'].astype(int)
    all_results = all_results.set_index(['depth', 'step'])
    all_results = all_results.sort_index(level=['depth', 'step'])
    return all_results

# %%
def convert_to_mean_std_string(df, col='loss.test', round_num=2):
    #Fernando
    #df = df[df['depth'].apply(lambda x: str(x).isdigit())]
    #df = df[df['step'].apply(lambda x: str(x).isdigit())]
    #Fernando
    std = df.groupby(['depth', 'step'], as_index=True).std().round(round_num)
    mean = df.groupby(['depth', 'step'], as_index=True).mean().round(round_num)
    string_it = mean[col].astype(str) + ' $\pm$ ' + std[col].astype(str)
    return mean, string_it

# %%
def extract_multiindex_step(df, steps=[1, 8, 32, 128]):
    return df.iloc[[True if x in [1, 8, 32, 128] else False for x in df.index.get_level_values(1)]]

# %% [markdown]
# # EigenWorms Table

# %%
# Get data

#model_name='hyperopt-odernn'
#ew = parse_results('UEA', 'EigenWorms', model_name, sort_key='test',average_over=None, print_frame=False, pretty_std=False)
folder = '/nfs/home/fernandom/github/neuralRDE/experiments/1-hyper_validation/sinusoidal/hyperopt-sinusoidal-ncde/Other/Sinusoidal/hyperopt-sinusoidal-ncde'
ew = parse_results(folder, 
                   None, 
                   None, 
                   sort_key='test',
                   average_over=None, 
                   print_frame=False, 
                   pretty_std=False)
ew['acc.test'] = (ew['acc.test'] * 100)
ew['acc.val'] = (ew['acc.val'] * 100)
ew['acc.train'] = (ew['acc.train'] * 100)
csv_directory = folder + '/summary_'+folder.split('/')[-1]+'.csv'
#ew.to_csv('/nfs/home/fernandom/github/neuralRDE/experiments/models_final/UEA/EigenWorms/'+model_name+'/summary_'+model_name+'.csv')
ew.to_csv(csv_directory)
# # Creating a BIDMC Results table

# %%
# Get data


#model_name='hyperopt'
folder = '/nfs/home/fernandom/github/neuralRDE/experiments/hyperopt-hr-odernn/TSR/BIDMC32HR/hyperopt-hr-odernn'
get_frame = lambda name: parse_results_bidmc(folder, 
                                       None, 
                                       '', 
                                       sort_key='test',
                                       average_over=None, 
                                       print_frame=False, 
                                       pretty_std=False)
#rr = get_frame('BIDMC32RR')
hr = get_frame('BIDMC32HR')
#hr.to_csv('/nfs/home/fernandom/github/neuralRDE/experiments/models_final/TSR/BIDMC32HR/'+model_name+'/summary_'+model_name+'.csv')
csv_directory = folder + '/summary_'+folder.split('/')[-1]+'.csv'
hr.to_csv(csv_directory)
