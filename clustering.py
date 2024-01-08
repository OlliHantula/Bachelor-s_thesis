import numpy as np
import anndata as ad
import scanpy as sc
import squidpy as sq
import pickle

import os
os.chdir("/lustre/scratch/krolha")

def qc_and_normalize(adata):
    # normalize and calculate leiden clustering
    sc.pp.filter_genes(adata, min_cells=5)
    sc.pp.filter_cells(adata, min_counts=500) 
    sc.pp.normalize_total(adata, exclude_highly_expressed=True)
    return adata

def spatially_aware_clustering(adata, seed=36345, proximity_weight=0.3,res=1.0):
    import squidpy as sq
    # Define the joint adjacency weighting
    sc.pp.scale(adata)
    sc.pp.highly_variable_genes(adata)
    sc.pp.pca(adata, n_comps=15)
    sc.pp.neighbors(adata, random_state=seed)
    sq.gr.spatial_neighbors(adata, n_rings=2, coord_type="grid", n_neighs=6,transform='cosine')
    joint_adj = adata.obsp['spatial_connectivities']*proximity_weight + adata.obsp['connectivities']
    sc.tl.leiden(adata,adjacency=joint_adj,key_added='joint_leiden_clusters',resolution=res,random_state=seed)
    return adata

def save_to_pickle(obj,filename):
    import pickle
    with open(filename, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_from_pickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

data = load_from_pickle("./individual_sections_normalized_clustered.pickle")
print(data)

adata = sc.read_visium('/lustre/scratch/krolha/spatial_results/PC_02_10136_VAS/outs')
qc_and_normalize(adata)
spatially_aware_clustering(adata, proximity_weight=0.1, res=0.7)
sq.pl.spatial_scatter(adata, color='joint_leiden_clusters', title=f"weight = {0.1}, res = {0.7}")
save_to_pickle(adata, "./bachelor/data/PC_02_10136_VAS_pw0.1_res0.7")

bdata = sc.read_visium('/lustre/scratch/krolha/spatial_results/PC_02_10136_VAS/outs')
qc_and_normalize(bdata)
spatially_aware_clustering(bdata, proximity_weight=0.3, res=1.0)
sq.pl.spatial_scatter(bdata, color='joint_leiden_clusters', title=f"weight = {0.3}, res = {1.0}")
save_to_pickle(bdata, "./bachelor/data/PC_02_10136_VAS_pw0.3_res1.0")
