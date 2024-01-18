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
from experiments.parse_results import parse_results

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

ew = parse_results('UEA', 
                   'EigenWorms', 
                   'hyperopt', 
                   sort_key='test',
                   average_over=None, 
                   print_frame=False, 
                   pretty_std=False)
ew['acc.test'] = (ew['acc.test'] * 100)

# %%
ew_mean, ew_string = convert_to_mean_std_string(ew, col='acc.test', round_num=1)
ew_mean['elapsed_time'] = (ew_mean['elapsed_time'] / (60 ** 2)).round(1)
ew_mean['memory_usage'] = ew_mean['memory_usage'].round(1)

ew_string = pd.concat((ew_string, ew_mean[['memory_usage', 'elapsed_time']]), axis=1)
ew_string = fix_index(ew_string)

# %%
ew_all = ew_string[['acc.test', 'elapsed_time', 'memory_usage']]
with open('tables/eigenworms_full.tex', 'w') as file:
    file.write(ew_all.to_latex(escape=False))

# %%
ew_table = extract_multiindex_step(ew_all)
with open('tables/eigenworms.tex', 'w') as file:
    file.write(ew_table.to_latex(escape=False))

# %% [markdown]
# # Creating a BIDMC Results table

# %%
# Get data
get_frame = lambda name: parse_results('TSR', 
                                       name, 
                                       'hyperopt', 
                                       sort_key='test',
                                       average_over=None, 
                                       print_frame=False, 
                                       pretty_std=False)
#rr = get_frame('BIDMC32RR')
hr = get_frame('BIDMC32HR')
#sp = get_frame('BIDMC32SpO2')

# %% [markdown]
# If we want just the mean of mem usage, time taken

# %%
#rr_mean, rr_string = convert_to_mean_std_string(rr)
hr_mean, hr_string = convert_to_mean_std_string(hr)
#sp_mean, sp_string = convert_to_mean_std_string(sp)

# %%
# Mean mem usage
get_mean = lambda col: pd.concat((rr_mean[col], hr_mean[col], sp_mean[col]), axis=1).mean(axis=1)
mean_mem = get_mean('memory_usage').astype(int)
mean_time = get_mean('elapsed_time').astype(int)

# %%
# Combine
all_results = pd.concat((rr_string, hr_string, sp_string, mean_mem, mean_time), axis=1)
all_results = fix_index(all_results)

# %%
with open('tables/mean_bidmctex.tex', 'w') as file:
    file.write(all_results.to_latex(escape=False))

# %% [markdown]
# Suppose we just wanted the full results

# %%
# All results
memory = pd.concat([x['memory_usage'] for x in [rr_mean, hr_mean, sp_mean]], axis=1).round(1)
time = (pd.concat([x['elapsed_time'] for x in [rr_mean, hr_mean, sp_mean]], axis=1) / (60 ** 2)).round(1)

# Combine
all_results_ = pd.concat((rr_string, hr_string, sp_string, memory, time), axis=1)
all_results_.reset_index(inplace=True)
all_results_['depth'] = all_results_['depth'].astype(int)
all_results_['step'] = all_results_['step'].astype(int)
all_results_.set_index(['depth', 'step'], inplace=True)
all_results_.sort_index(level=['depth', 'step'], inplace=True)

# %%
# Mean the memory usage
all_results_['mean_memory_usage'] = all_results_['memory_usage'].mean(axis=1).round(1).values
all_results_.drop('memory_usage', axis=1, inplace=True);

# %%
all_results_ = all_results_.iloc[[False if x in [20, 50] else True for x in all_results_.index.get_level_values(1)]]

# %%
output = all_results_.iloc[[True if x in [1, 8, 32, 128] else False for x in all_results_.index.get_level_values(1)]]

# %%
with open('full_bidmctex.tex', 'w') as file:
    file.write(output.to_latex(escape=False))

# %%
with open('tables/bidmc_full.tex', 'w') as file:
    file.write(all_results_.to_latex(escape=False))

# %%
all_results_

# %%



