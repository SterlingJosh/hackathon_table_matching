import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

from dataset import prepare_data
import timeit


REGENERATE_DATA=False
DATA_LOC = 'stats'
COLUMNS_FOR_HISTREF = ['CreationDate', 'LastAccessDate']
PARENT_HISTREF = 'CreationDate'
CHILD_HISTREF = 'histref_CreationDate'

def choose_neighbors(idcs, distances, num_neighbors):

    for i in range(1,num_neighbors):
            # check if first column contains duplicates. Replace duplicates with column i
            col = idcs[:,0]
            _, unique_idcs = np.unique(col, return_index=True)
            result = col == col
            result[unique_idcs] = False
            is_duplicate = np.invert(result)
            idcs[is_duplicate,0] = idcs[is_duplicate,i]
            distances[is_duplicate, 0] = distances[is_duplicate,i]

def duplicate_table(df):
    """returns empty copy of dataframe"""   
    df_new = df.copy(deep=False)[0:0]
    return df_new.astype(df.dtypes.to_dict())

def do_baseline(parent, child):
    # do a baseline
    p = parent[PARENT_HISTREF].values
    c = child[CHILD_HISTREF].values
    p.sort()
    c.sort()
    diff = np.mean(np.abs(p - c))
    print(f"for comparison: baseline error of sorting merge = {diff}")


def summarize_results(count, parent_new, og_parent_len, dists, parent_idcs, dist_list, runtime_distances, ax, fig):
    
        print(f"it {count}. rows in new parent {len(parent_new)}. \
            rows to go {og_parent_len - len(parent_new)}. \
            Added {len(parent_idcs)} rows, with mean distance {np.mean(dists)}. \
                Total average = {np.mean(dists)} \
                max distance = {np.max(dists)}. 95th percentile {np.percentile(dists, q=95)}")

        ax[0].hist(dists, bins=40, range=[0, 55], color='darkblue')
        ax[0].set_title('distribution of distances',fontsize=12)
        ax[0].set_xlabel('distance (days)')
        ax[0].set_ylabel('count (rows)')
      
        # plot 
        ax[1].plot(runtime_distances['progress'], color='darkblue')
        ax[1].set_title(f'progress over time',fontsize=12)
        ax[1].set_xlabel('time')
        ax[1].set_ylabel('progress (percent complete)')

        ax[2].plot(runtime_distances['batch_distances'], color='darkblue')
        ax[2].set_title(f'batch distance over time',fontsize=12)
        ax[2].set_xlabel('time')
        ax[2].set_ylabel('distance')

        plt.subplots_adjust(hspace = 0.5)

        plt.savefig(f'a_plots/imgs/subplots_{count}.png')


def pair_tables(child, parent):
    """pairs two tables by shuffling them, based on minimizing the distance between histref columns

    Args:
        child (df): child table
        parent (df): parent table
    """

    print("merge_tables")

    parent = parent.reset_index()
    child = child.reset_index()

    # make new empty tables
    parent_new = duplicate_table(parent)
    child_new = duplicate_table(child)

    dist_list = []
    
    # plot distributions
    plt.style.use('seaborn-deep')
    plt.xlabel('date (days from 1900)')
    plt.hist(child[CHILD_HISTREF], bins=40, alpha=0.5, label='child')
    plt.hist(parent[PARENT_HISTREF], bins=40, alpha=0.5, label='parent')
    plt.title('comparison of date distributions parent and child')
    plt.xlabel('distance (days)')
    plt.ylabel('count (rows)')
    plt.legend(loc='upper right')
    plt.show()

    count = 0
    num_neighbors = 8
    noise_beta = 5
    num_children_to_make = len(child)
    og_parent_len = len(parent)

    do_baseline(parent, child)
    
    # for plotting
    start = timeit.default_timer()
    runtime_distances = {'runtimes': [start], 'batch_distances': [0], 'progress':[0]}

    fig, ax = plt.subplots(3, figsize=(8, 12), gridspec_kw={'height_ratios': [2,1,1]})

    # now continuously constructing and updating the kd-tree and querying for closest matches
    while(len(child_new) < num_children_to_make and len(child) > num_neighbors and len(parent) > num_neighbors and count < 100):

        # adding random noise to avoid oversampling the same indices
        parent_vals = np.expand_dims(parent[PARENT_HISTREF].values  + (np.random.rand(len(parent)) - 0.5)*noise_beta , axis=1) #  
        child_vals = np.expand_dims(child[CHILD_HISTREF].values + (np.random.rand(len(child)) - 0.5)*noise_beta , axis=1)  # + 

        # get the nearest parent date (idx) for each child date
        tree = KDTree(child_vals, leaf_size=40)
        print('querying')
        distances, idcs = tree.query(parent_vals, k=num_neighbors, dualtree=True, return_distance=True) 
       
        # for each of n nearest neighbors, set most uniques to the first column        
        # choose_neighbors(idcs, distances, num_neighbors)
        child_idcs, parent_idcs = np.unique(idcs[:,0], return_index=True)
        
        # add matches to new dataframes
        parent_new = parent_new.append(parent.iloc[parent_idcs])
        child_new = child_new.append(child.iloc[child_idcs])
        # remove matches from old dataframes
        child.drop(child.index[child_idcs], inplace=True)
        parent.drop(parent.index[parent_idcs], inplace=True)

        count +=1
        # index only the selected distances
        dists = distances[:,0]
        dist_list.extend(dists)

        runtime_distances['runtimes'].append(timeit.default_timer() - start)
        runtime_distances['batch_distances'].append(np.mean(dists))
        runtime_distances['progress'].append((len(parent_new) / og_parent_len)*100)
        
        # print, plot and save results
        summarize_results(count, parent_new, og_parent_len, dists, parent_idcs, dist_list, runtime_distances, ax, fig)
   
    # for the last part we're left with a few rows. These will be coppied irrespective of match quality...
    if len(child_new) < num_children_to_make:
        print("short on rows")
        child_new = child_new.append(child)
        parent_new = parent_new.append(parent)

    total_distance_80 = np.mean(abs(child_new[CHILD_HISTREF].values.astype(np.float32)[:round(len(child_new)*0.8)] - parent_new[PARENT_HISTREF].values.astype(np.float32)[:round(len(parent_new)*0.8)]))
    total_distance_100 = np.mean(abs(child_new[CHILD_HISTREF].values.astype(np.float32) - parent_new[PARENT_HISTREF].values.astype(np.float32)))
    print(f"merge complete. Total distance of best 80% = {round(total_distance_80,2)}. total distance of 100% = { round(total_distance_100,2)}")


if __name__ == "__main__":

    if (REGENERATE_DATA):
        df_post, df_user = prepare_data(COLUMNS_FOR_HISTREF, DATA_LOC)

        df_post.to_pickle('df_post_processed.pickle')
        df_user.to_pickle('df_user_processed.pickle')
    else:
        df_post = pd.read_pickle('df_post_processed.pickle')
        df_user = pd.read_pickle('df_user_processed.pickle')

    pair_tables(df_post, df_user)
