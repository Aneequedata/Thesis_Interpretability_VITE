import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import graphviz
import os
import warnings
import time
import re
from copy import deepcopy
warnings.filterwarnings("ignore")
import xgboost as xgb

def load_data(data_folder,filename,subset=None):
    '''
    :param data_folder: folder where there are the data
    :param filename: name of the csv file of the dataframe
    :param subset: list of subset columns to use
    '''
    file = os.path.join(data_folder,filename)
    df = pd.read_csv(file)
    if subset!=None:
        df = df[subset]
    return df

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def proximityMatrix(model, X, learning_rate, normalize=True):
    start_time = time.time()
    print("Starting proximityMatrix function")

    # Get the underlying Booster object
    booster = model.get_booster()

    # Determine the number of boosting rounds
    try:
        # Use best_iteration + 1 to get all iterations if available
        num_boost_round = model.best_iteration + 1
    except AttributeError:
        # Use the number of boosting rounds from the booster object
        num_boost_round = booster.num_boosted_rounds()

    # Ensure num_boost_round is an integer and not a method
    if not isinstance(num_boost_round, int):
        num_boost_round = int(num_boost_round)
    print(f"Number of boosting rounds: {num_boost_round}")

    # Iterate over the trees and obtain the prediction for each tree
    tree_predictions = []
    print("Starting tree prediction iteration")
    for tree_index in range(1, num_boost_round + 1):
        iter_start_time = time.time()
        # Predict up to the current tree_index
        tree_prediction = booster.predict(
            data=xgb.DMatrix(X),
            iteration_range=(0, tree_index)
        )
        tree_predictions.append(tree_prediction)
        iter_end_time = time.time()
        print(f"Tree index {tree_index}: prediction took {iter_end_time - iter_start_time:.2f} seconds")

    print("Tree predictions completed")
    # Convert the list of tree predictions into a numpy array
    tree_predictions = np.array(tree_predictions)
    tree_predictions = pd.DataFrame(tree_predictions)

    # Calculate weights
    print("Calculating weights")
    weights = [np.var(0.5 - sigmoid(tree_predictions.iloc[0])) * learning_rate ** 2]
    for t in range(1, num_boost_round):
        weight = np.var((sigmoid(tree_predictions.iloc[t]) - sigmoid(tree_predictions.iloc[t - 1])) / learning_rate)
        weights.append(weight * learning_rate ** 2)
    weights = np.array(weights)
    weights = weights / sum(weights)

    # Apply trees to X to get leaf indices
    terminals = model.apply(X)  # Directly use X without wrapping it in xgb.DMatrix
    nTrees = num_boost_round
    print(f"Number of trees (nTrees): {nTrees}")

    a = terminals[:, 0]
    elem = np.equal.outer(a, a)
    proxMat = 1 * elem
    proxMat_w = weights[0] * elem

    print("Starting proximity matrix calculations")
    for i in range(1, nTrees):
        iter_start_time = time.time()
        a = terminals[:, i]
        elem = np.equal.outer(a, a)
        proxMat_w += weights[i] * elem
        proxMat += 1 * elem
        iter_end_time = time.time()
        print(f"Tree index {i}: proximity matrix calculation took {iter_end_time - iter_start_time:.2f} seconds")


    if normalize:
        proxMat = proxMat / nTrees
        proxMat_w = proxMat_w  # No need to divide by nTrees since weights already normalize it

    proxMat_w = pd.DataFrame(proxMat_w)
    proxMat = pd.DataFrame(proxMat)
    weights = pd.DataFrame(weights)

    end_time = time.time()
    print(f"Proximity matrix calculations completed in {end_time - start_time:.2f} seconds")

    return proxMat, proxMat_w, weights


class VisualizationRF:

    def __init__(self,trained_model,data,max_depth,n_estimators):

        self.model = trained_model
        self.data = data
        self.max_depth = max_depth
        self.list_features = list(self.data.columns)[:-1]
        self.n_estimators = n_estimators

    def change_df_xgb(self,df_xgb):
        for t in list(df_xgb['Tree'].unique()):
            v = list(df_xgb[df_xgb['Tree'] == t]['Node'].values)
            comp = list(range(len(v)))

            el = -1
            if v != comp:
                for elem in comp:
                    if elem not in v:
                        el = elem
                        break

                dif_d = {v[i]: i for i in range(el, comp[-1] + 1)}


                df_xgb.loc[df_xgb[df_xgb['Tree'] == t]['Node'].apply(
                    lambda x: dif_d[x] if x in dif_d.keys() else x).index, 'Node'] = df_xgb[df_xgb['Tree'] == t]['Node'].apply(
                    lambda x: dif_d[x] if x in dif_d.keys() else x)

                df_xgb.loc[df_xgb[df_xgb['Tree'] == t]['ID'].apply(
                    lambda x: str(t) + '-' + str(dif_d[int(x.split('-')[1])]) if int(
                        x.split('-')[1]) in dif_d.keys() else x).index, 'ID'] = \
                    df_xgb[df_xgb['Tree'] == t]['ID'].apply(
                        lambda x: str(t) + '-' + str(dif_d[int(x.split('-')[1])]) if int(
                            x.split('-')[1]) in dif_d.keys() else x)

                df_xgb.loc[df_xgb[df_xgb['Tree'] == t]['Yes'].apply(
                    lambda x: str(t) + '-' + str(dif_d[int(x.split('-')[1])]) if str(x) != 'nan' and int(
                        x.split('-')[1]) in dif_d.keys() else x).index, 'Yes'] = \
                    df_xgb[df_xgb['Tree'] == t]['Yes'].apply(
                        lambda x: str(t) + '-' + str(dif_d[int(x.split('-')[1])]) if str(x) != 'nan' and int(
                            x.split('-')[1]) in dif_d.keys() else x)

                df_xgb.loc[df_xgb[df_xgb['Tree'] == t]['No'].apply(
                    lambda x: str(t) + '-' + str(dif_d[int(x.split('-')[1])]) if str(x) != 'nan' and int(
                        x.split('-')[1]) in dif_d.keys() else x).index, 'No'] = \
                    df_xgb[df_xgb['Tree'] == t]['No'].apply(
                        lambda x: str(t) + '-' + str(dif_d[int(x.split('-')[1])]) if str(x) != 'nan' and int(
                            x.split('-')[1]) in dif_d.keys() else x)

                df_xgb.loc[df_xgb[df_xgb['Tree'] == t]['Missing'].apply(
                    lambda x: str(t) + '-' + str(dif_d[int(x.split('-')[1])]) if str(x) != 'nan' and int(
                        x.split('-')[1]) in dif_d.keys() else x).index, 'Missing'] = \
                    df_xgb[df_xgb['Tree'] == t]['Missing'].apply(
                        lambda x: str(t) + '-' + str(dif_d[int(x.split('-')[1])]) if str(x) != 'nan' and int(
                            x.split('-')[1]) in dif_d.keys() else x)
        return df_xgb




    def heatmap_RF_featuredepth(self,df_perc,directory='figure',show=None):
        '''
        :param df_perc: df #features X #depth that say how many time each feature is used at each depth, in percentage
        :param directory: directory where to save the heatmap
        :param show: if True, the heatmap is displayed
        :return: an heatmap to visualize the features that are more used, at each depth, in the whole forest
        '''
        # HEATMAP
        fig, ax = plt.subplots(figsize=(len(df_perc.index)*.5,len(df_perc.index)*.5))

        # Add title to the Heat map
        title = "XGB Heatmap with % of features usage per level"

        # Set the font size and the distance of the title from the plot
        plt.title(title,fontsize=13)
        ttl = ax.title
        ttl.set_position([0.5,1.05])

        # Hide ticks for X & Y axis
        ax.set_xticks([])
        ax.set_yticks([])

        # Remove the axes
        #sns.heatmap(df_perc,annot =np.array(df_perc.values*100,dtype='int'),fmt="",cmap="YlGnBu",linewidths=0.30,ax=ax)
        map = sns.color_palette("Blues", as_cmap=True)

        sns.heatmap(df_perc, annot=np.array(df_perc.values * 100, dtype='float'), fmt=".1f", cmap=map, linewidths=0.30,
                    ax=ax)
        if show ==True:
            # Display the Heatmap
            plt.show()

        #fig.savefig(path+"heat_map.png")
        fig.savefig(os.path.join(os.getcwd(),directory,"heat_map.png"),bbox_inches="tight")
        plt.close()

    def feature_usage_level(self, weights=None):
        # Initialization and reference DataFrame creation
        n = [1]
        for iter in range(self.max_depth):
            n.append(n[-1] * 2)
        num_nodes = sum(n[:-1])
        num_nodes_tot = sum(n)
        depths = [i for i in range(self.max_depth + 1)]

        dict_ref = {'Node': [i for i in range(num_nodes)], 'Yes': [], 'No': [], 'depth': []}
        for i in range(1, num_nodes_tot, 2):
            dict_ref['Yes'].append(i)
            dict_ref['No'].append(i + 1)
        for i in range(num_nodes, num_nodes_tot):
            dict_ref['Node'].append(i)
            dict_ref['Yes'].append('nan')
            dict_ref['No'].append('nan')
        for i in range(len(depths)):
            dict_ref['depth'] += [depths[i]] * n[i]

        df_ref = pd.DataFrame.from_dict(dict_ref)

        df_xgb = self.model.get_booster().trees_to_dataframe()
        df_xgb['Feature'] = df_xgb['Feature'].apply(
            lambda x: 'f' + str(''.join([char for char in x if char.isdigit()])) if x != 'Leaf' else x)
        df_xgb = self.change_df_xgb(df_xgb)
        df_xgb['Yes'] = df_xgb['Yes'].apply(lambda x: str(x))
        df_xgb['No'] = df_xgb['No'].apply(lambda x: str(x))

        # Align nodes with the reference DataFrame
        for t in range(self.n_estimators):
            tree_ = t
            changed = []
            for node in range(len(df_ref)):
                nod_ref = node
                if node in changed:
                    node = str(node) + '_'
                if node in df_xgb[df_xgb['Tree'] == tree_]['Node'].values:
                    if df_xgb[(df_xgb['Tree'] == tree_) & (df_xgb['Node'] == node)]['Yes'].values[0] == 'nan' or \
                            int(df_xgb[(df_xgb['Tree'] == tree_) & (df_xgb['Node'] == node)]['Yes'].values[0].split(
                                '-')[1]) == df_ref[df_ref['Node'] == nod_ref]['Yes'].values[0]:
                        continue
                    else:
                        subs = {
                            int(df_xgb[(df_xgb['Tree'] == tree_) & (df_xgb['Node'] == node)]['Yes'].values[0].split(
                                '-')[1]): df_ref[df_ref['Node'] == nod_ref]['Yes'].values[0],
                            int(df_xgb[(df_xgb['Tree'] == tree_) & (df_xgb['Node'] == node)]['No'].values[0].split('-')[
                                    1]): df_ref[df_ref['Node'] == nod_ref]['No'].values[0]}
                        changed += list(subs.values())
                        df_xgb.loc[df_xgb[(df_xgb['Tree'] == tree_) & (df_xgb['Node'] == node)].index[0], 'Yes'] = str(
                            tree_) + '-' + str(subs[list(subs.keys())[0]])
                        df_xgb.loc[df_xgb[(df_xgb['Tree'] == tree_) & (df_xgb['Node'] == node)].index[0], 'No'] = str(
                            tree_) + '-' + str(subs[list(subs.keys())[1]])
                        for key in subs.keys():
                            if list(df_xgb[(df_xgb['Tree'] == tree_) & (df_xgb['Node'] == key)].values) != []:
                                df_xgb.loc[
                                    df_xgb[(df_xgb['Tree'] == tree_) & (df_xgb['Node'] == key)].index[0], 'Node'] = str(
                                    subs[key]) + '_'
                            else:
                                df_xgb.loc[df_xgb[(df_xgb['Tree'] == tree_) & (df_xgb['Node'] == str(key) + '_')].index[
                                    0], 'Node'] = str(subs[key]) + '_'

        df_xgb['Node'] = df_xgb['Node'].apply(lambda x: str(x))

        # Inspect the 'Node' column to identify unique values
        print("Unique values in 'Node':", df_xgb['Node'].unique())

        # Handle 'nan_' and clean up 'Node' column
        df_xgb['Node'] = df_xgb['Node'].replace('nan_', np.nan)
        df_xgb.dropna(subset=['Node'], inplace=True)

        # Apply transformation to remove underscores and convert to integers
        df_xgb['Node'] = df_xgb['Node'].apply(lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else np.nan)

        # Handle any remaining NaNs
        df_xgb.dropna(subset=['Node'], inplace=True)

        # Verify changes
        print(df_xgb)

        # Continue with merging and further processing
        df_xgb = df_xgb.merge(df_ref[['Node', 'depth']], on='Node')
        df_xgb.sort_values(['Tree', 'Node'], inplace=True)
        df_xgb.reset_index(drop=True, inplace=True)
        df_xgb['Feature'] = df_xgb['Feature'].apply(
            lambda x: int(x.split('f')[1]) if x.split('f')[1].isdigit() else 'Leaf')

        grouped_df = df_xgb[['depth', 'Feature', 'Tree']].groupby(['depth', 'Feature']).size().reset_index(name='count')
        pivot_df = grouped_df.pivot(index='Feature', columns='depth', values='count')
        pivot_df = pivot_df.applymap(lambda x: str(x))
        pivot_df = pivot_df.applymap(lambda x: int(float(x)) if x != 'nan' else 0)
        pivot_df.drop('Leaf', inplace=True)
        pivot_df.drop(columns=[pivot_df.columns[-1]], inplace=True)
        pivot_df.reset_index(inplace=True)

        for i in range(len(self.list_features)):
            if i not in pivot_df['Feature'].values:
                new_row = {**{'Feature': i}, **{l: 0 for l in range(self.max_depth)}}
                new_row_df = pd.DataFrame([new_row])
                pivot_df = pd.concat([pivot_df, new_row_df], ignore_index=True)

        df_depth = pivot_df
        df_depth.sort_values('Feature', inplace=True)
        df_depth.reset_index(drop=True, inplace=True)
        df_depth.set_index('Feature', inplace=True)
        df_depth.rename(columns={i: 'depth_' + str(i) for i in range(self.max_depth)}, inplace=True)
        df_perc = df_depth / df_depth.sum(axis=0)
        df_perc.rename(columns={'depth_' + str(ix): 'level_' + str(ix) for ix in list(range(self.max_depth))},
                       inplace=True)
        df_perc.rename(index={i: self.list_features[i] for i in range(len(self.list_features))}, inplace=True)

        group_df = df_xgb[['Node', 'Feature', 'Tree']].groupby(['Node', 'Feature']).size().reset_index(name='count')
        df_nodes = group_df.pivot(index='Feature', columns='Node', values='count')
        df_nodes = df_nodes.applymap(lambda x: str(x))
        df_nodes = df_nodes.applymap(lambda x: int(float(x)) if x != 'nan' else 0)
        df_nodes.drop('Leaf', inplace=True)
        col_list = [el for el in list(range(num_nodes_tot - n[-1], num_nodes_tot)) if el in df_nodes.columns]
        df_nodes.drop(columns=col_list, inplace=True)
        df_nodes.reset_index(inplace=True)

        for i in range(len(self.list_features)):
            if i not in df_nodes['Feature'].values:
                new_row = {**{'Feature': i}, **{l: 0 for l in range(num_nodes_tot - n[-1])}}
                new_row_df = pd.DataFrame([new_row])
                df_nodes = pd.concat([df_nodes, new_row_df], ignore_index=True)

        df_nodes.sort_values('Feature', inplace=True)
        df_nodes.reset_index(drop=True, inplace=True)
        df_nodes.set_index('Feature', inplace=True)
        df_nodes.rename(index={i: self.list_features[i] for i in range(len(self.list_features))}, inplace=True)
        df_nodes.rename(columns={i: 'node_' + str(i) for i in range(n[-1])}, inplace=True)
        df_nodes_perc = df_nodes.div(df_nodes.sum(axis=0), axis=1)

        if str(type(weights)) != "<class 'NoneType'>":
            group_df = df_xgb[['Tree', 'Feature', 'depth']].groupby(['Tree', 'depth', 'Feature']).size().reset_index(
                name='count')
            weights.reset_index(inplace=True)
            weights = weights.rename(columns={'index': 'Tree'})
            group_df = group_df.merge(weights, on='Tree')
            group_df['Freq'] = group_df[0] * group_df['count']
            weight_times = group_df[['Feature', 'depth', 'Freq', 'count']].groupby(['Feature', 'depth']).sum()
        else:
            weight_times = False

        return df_xgb, df_depth, df_perc, df_nodes, df_nodes_perc, weight_times



    def heatmap_nodes(self,df_nodes_perc,directory,df_xgb):
        '''
        :param df_nodes_perc: dataframe that have as index the features and columns the nodes of the tree
        in the df is saved the number of time each feature is used in each node
        :param directory: directory where to save the images
        :param tree_dict: dictionary of dictionaries, with keys the name of the tree and values a dictionary with
        keys the integer of the node and value a list with [depth of the node,feature used in the node,threshold uded
        in that node]
        :param max_depth: depth max of the trees
        :param features_name: list with the name of the features
        :return: the images of the heatmap to save at each node
        '''


        node_name = list(df_nodes_perc.columns)

        for name in node_name:
            num_node = int(name.split('_')[1])
            # HEATMAP
            fig, ax = plt.subplots(figsize=(len(df_nodes_perc.index)*.5*0.5,len(df_nodes_perc.index)*.3*0.2))#,len(df_nodes_perc.index)*2 ))
            fig.set_tight_layout(True)
            # Add title to the Heat map
            title = "_%"
            # Set the font size and the distance of the title from the plot
            plt.title(title, fontsize=18)
            ttl = ax.title
            ttl.set_position([0.5, 1.05])
            # Hide ticks for X & Y axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set(ylabel=None)


            df_torep = pd.DataFrame(df_nodes_perc[name]).T
            df_torep = df_torep.iloc[:, list(np.where(df_torep.values[0] != 0)[0])]
            df_xgb['Split'] = df_xgb['Split'].apply(lambda x: str(x))
            for f in list(df_torep.columns):
                if (df_xgb[(df_xgb['Node']==int(name.split('_')[1]))&(df_xgb['Feature']==self.list_features.index(f))]['Split'].values!='nan').any():
                    vals = [float(v) for v in df_xgb[(df_xgb['Node']==int(name.split('_')[1]))&(df_xgb['Feature']==self.list_features.index(f))]['Split'].values]
                    inf = round(min(vals),2)
                    sup = round(max(vals),2)
                    if inf ==sup:
                        df_torep.rename(columns={f:f+' '+str(inf)
                                         }, inplace=True)
                    else:
                        df_torep.rename(columns={f: f + ' ' + '['+str(inf)+','+str(sup)+']'
                                                 }, inplace=True)
            df_torep.rename(index={name:''},inplace=True)
            # Remove the axes
            map = sns.color_palette("Blues", as_cmap=True)
            sns.heatmap(df_torep, annot=np.array(df_torep.values * 100, dtype='float'), fmt=".1f", cmap=map, linewidths=0.30,
                        ax=ax)

            fig.savefig(os.path.join(os.getcwd(),directory,"heatmap_"+name+".png"),bbox_inches="tight")
            plt.close(fig)



    def tree_heatmap(self,filename,df_nodes_perc,directory):
        '''
        function to represent the tree with heatmap at each node
        :param filename: name of the pdf file where to save the file
        :param df_nodes_perc: dataframe with the percentage with which each feature is used in a given node
        :param max_depth: maximum depth of the tree in the random forest
        :param directory: directory where to save the images
        :return: a pdf file with the tree with heatmap
        '''

        n = [1]  # inizializing the list that indicates the number of nodes at each depth
        for iter in range(self.max_depth):
            n.append(n[-1] * 2)
        num_nodes = sum(
            n[:-1])  # number of nodes of the representation ( the last depth is not represented since there is
        # not a split in the last level of the tree)
        num_nodes_tot = sum(n)
        depths = [i for i in range(self.max_depth+1)]

        dict_ref = {'Node':[i for i in range(num_nodes)],'Yes':[],'No':[],'depth':[]}
        for i in range(1,num_nodes_tot,2):
            dict_ref['Yes'].append(i)
            dict_ref['No'].append(i+1)
        for i in range(num_nodes,num_nodes_tot):
            dict_ref['Node'].append(i)
            dict_ref['Yes'].append('nan')
            dict_ref['No'].append('nan')
        for i in range(len(depths)):
            dict_ref['depth']+=[depths[i]]*n[i]

        df_ref = pd.DataFrame.from_dict(dict_ref)

        f = graphviz.Digraph(filename= filename,format='png')
        df_support = df_nodes_perc.dropna(axis=1)
        names = list(df_support.columns)

        positions = ['' for i in range(len(names))]
        for name, position in zip(names, positions):
            f.node(name, position,
                   image=os.path.join(os.getcwd(),directory,'nodes','heatmap_'+name+'.png'),shape="plaintext")

        df_ref_m = df_ref[['Node','depth']][df_ref['depth']!=self.max_depth].set_index('Node')
        df_ref_dict = df_ref_m.to_dict()['depth']
        node_depth = {}
        for key,value in df_ref_dict.items():
            node_depth['node_'+str(key)]=value

        #node_depth = {f"node_{nd}": int(key.split("_")[1]) for key, values in self.node_depth_dict().items() if
        #              key != f"d_{self.max_depth}" for nd in values}
        #df_nod = df_ref[(df_ref['depth']!=self.max_depth)&(df_ref['depth']!=self.max_depth-1)]
        #for node in df_nod['Node'].values:
        #    f.edge('node_'+str(df_nod['Node']), 'node_'+str(df_nod['Yes']))
        #    f.edge('node_' + str(df_nod['Node']), 'node_' + str(df_nod['No']))

        for d in range(1,self.max_depth):
            n_p0 = [ k for k,v in node_depth.items() if v==d-1]
            n_p1 = [ k for k,v in node_depth.items() if v==d]
            for elem in n_p0:
                if len(n_p0)!=0:
                    for i in range(2):
                        f.edge(elem,n_p1[i])
                    n_p0 = n_p0[1:]
                    n_p1 = n_p1[2:]

        f.graph_attr['dpi'] = '300'
        f.render(directory=os.path.join(os.getcwd(),directory), view=True,format='png').replace('\\', '/')






