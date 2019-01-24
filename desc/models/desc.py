"""
Keras implementation for Deep Embedded Clustering (DEC) algorithm for single cell analysis:

        Xiangjie Li et al 

Usage:
    use `python DEC.py -h` for help.

Author:
    Xiangjie Li. 2018.5.8
"""

from time import time as get_time

import math
import numpy as np
import random
import tensorflow as tf
import keras.backend as kb
import scanpy.api as sc
import pandas as pd
import os
import multiprocessing
from anndata import AnnData
from keras.engine.topology import Layer, InputSpec
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import SGD
#from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping
from sklearn.cluster import KMeans
from math import ceil
from time import time as get_time
from ..utilities import _get_max_prob

def getdims(x=(10000,200)):
    """
    This function will give the suggested nodes for each encoder layer
    return the dims for network
    """
    assert len(x)==2
    n_sample=x[0]
    if n_sample>20000:# may be need complex network
        dims=[x[-1],128,32]
    elif n_sample>10000:#10000
        dims=[x[-1],64,32]
    elif n_sample>5000: #5000
        dims=[x[-1],32,16] #16
    elif n_sample>2000:
        dims=[x[-1],128]
    elif n_sample>500:
        dims=[x[-1],64]
    else:
        dims=[x[-1],16]
    return dims

    
def train_single(data,dims=None,
        alpha=1.0,
        tol=0.005,
        init='glorot_uniform',
        n_clusters=None,
        louvain_resolution=1.0,
        n_neighbors=15,
        pretrain_epochs=300,
        batch_size=256,
        activation='relu',
        actincenter='tanh',
        drop_rate_SAE=0.2,
        is_stacked=True,
        use_earlyStop=True,
        use_ae_weights=False,
    save_encoder_weights=False,
        save_encoder_step=4,
        save_dir='result_tmp',
        max_iter=1000,
        epochs_fit=5, 
        num_Cores=20,
        num_Cores_tsne=10,
        use_GPU=True,
        random_seed=201809,
        verbose=True,
    do_tsne=False,
    learning_rate=150,
    perplexity=30,
        do_umap=False,
):
    if isinstance(data,AnnData):
        adata=data
    else:
        adata=sc.AnnData(data)
    #make sure dims 
    if dims is None:
        dims=getdims(adata.shape)
    assert dims[0]==adata.shape[-1],'the number of columns of x doesnot equal to the first element of dims, we must make sure that dims[0]==x.shape[0]'
    #if use_GPU and tf.test.is_gpu_available():
#set seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    total_cpu=multiprocessing.cpu_count()
    num_Cores=int(num_Cores) if total_cpu>int(num_Cores) else int(math.ceil(total_cpu/2)) 
    print('The number of cpu in your computer is',total_cpu)
    if use_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        #os.environ['CUDA_VISIBLE_DEVICES']=kb.tensorflow_backend._get_available_gpus()[0][-1]#use first GPUid
    else:
        #set only use cpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        kb.set_session(tf.Session(graph=tf.get_default_graph(),config=tf.ConfigProto(intra_op_parallelism_threads=num_Cores, inter_op_parallelism_threads=num_Cores)))
    if not use_ae_weights and os.path.isfile(os.path.join(save_dir,"ae_weights.h5")):
        os.remove(os.path.join(save_dir,"ae_weights.h5"))
  
    tic=get_time()#recored time         
    desc=DescModel(dims=dims,
              x=adata.X,
              alpha=alpha,
              tol=tol,
              init=init,
              n_clusters=n_clusters,
              louvain_resolution=louvain_resolution,
              n_neighbors=n_neighbors,
              pretrain_epochs=pretrain_epochs,
              epochs_fit=epochs_fit,
              batch_size=batch_size,
              random_seed=random_seed,
              activation=activation,
              actincenter=actincenter,
              drop_rate_SAE=drop_rate_SAE,
              is_stacked=is_stacked,
              use_earlyStop=use_earlyStop,
              use_ae_weights=use_ae_weights,
          save_encoder_weights=save_encoder_weights,
              save_encoder_step=save_encoder_step,
              save_dir=save_dir
    )
    desc.compile(optimizer=SGD(0.01,0.9),loss='kld')
    Embeded_z,q_pred=desc.fit(maxiter=max_iter,)
    print("The desc has been trained successfully!!!!!!")
    if verbose:
        print("The summary of desc model is:")
        desc.model.summary()
    print("The runtime of (resolution="+str(louvain_resolution)+")is:",get_time()-tic)
    y_pred=pd.Series(np.argmax(q_pred,axis=1),index=adata.obs.index,dtype='category')
    y_pred.cat.categories=list(range(len(y_pred.unique())))
    adata.obs['desc_'+str(louvain_resolution)]=y_pred
    adata.obsm['X_Embeded_z'+str(louvain_resolution)]=Embeded_z
    if do_tsne:
        num_Cores_tsne=int(num_Cores_tsne) if total_cpu>int(num_Cores_tsne) else int(math.ceil(total_cpu/2))
        sc.tl.tsne(adata,use_rep="X_Embeded_z"+str(louvain_resolution),learning_rate=learning_rate,perplexity=perplexity,n_jobs=num_Cores_tsne)
        adata.obsm["X_tsne"+str(louvain_resolution)]=adata.obsm["X_tsne"].copy()
        sc.logging.msg(' tsne finished', t=True, end=' ', v=4)
        sc.logging.msg('and added\n'
                 '    \'X_tsne\''+str(louvain_resolution),'the tsne coordinates (adata.obs)\n', v=4)
    if do_umap:
        sc.pp.neighbors(adata,n_neighbors=n_neighbors,use_rep="X_Embeded_z"+str(louvain_resolution)) 
        sc.tl.umap(adata)
        adata.obsm["X_umap"+str(louvain_resolution)]=adata.obsm["X_umap"].copy() 
        sc.logging.msg(' umap finished', t=True, end=' ', v=4)
        sc.logging.msg('and added\n'        
                 '    \'X_umap\''+str(louvain_resolution),'the umap coordinates (adata.obsm)\n', v=4)
        del adata.uns["neighbors"]

   
    #prob_matrix
    adata.uns['prob_matrix'+str(louvain_resolution)]=q_pred
    return adata


def train(data,dims=None,
        alpha=1.0,
        tol=0.005,
        init='glorot_uniform',
        n_clusters=None,
        louvain_resolution=[0.6,0.8],
        n_neighbors=10,
        pretrain_epochs=300,
        batch_size=256,
        activation='relu',
        actincenter='tanh',
        drop_rate_SAE=0.2,
        is_stacked=True,
        use_earlyStop=True,
        use_ae_weights=True,
    save_encoder_weights=False,
        save_encoder_step=5,
        save_dir='result_tmp',
        max_iter=1000,
        epochs_fit=5,
        num_Cores=20,
        num_Cores_tsne=10,
        use_GPU=False,
        random_seed=201809,
        verbose=True,
    do_tsne=False,
    learning_rate=150,
    perplexity=30,
        do_umap=False
): 
    """ Deep Embeded single cell clustering(DESC) API
    Conduct clustering for single cell data given in the anndata object or np.ndarray,sp.sparmatrix,or pandas.DataFrame
      
    
    Argument:
    ------------------------------------------------------------------
    data: :class:`~anndata.AnnData`, `np.ndarray`, `sp.spmatrix`,`pandas.DataFrame`
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes.
    dims: `list`, the number of node in encoder layer, which include input dim, that is
    [1000,64,32] represents for the input dimension is 1000, the first hidden layer have 64 node, and second hidden layer(or bottle neck layers) have 16 nodes. if not specified, it will be decided automatically according to the sample size.
    
    alpha: `float`, optional. Default: `1.0`, the degree of t-distribution.
    tol: `float`, optional. Default: `0.005`, Stop criterion, clustering procedure will be stoped when the difference ratio betwen the current iteration and last iteration larger than tol.
    init: `str`,optional. Default: `glorot_uniform`.
        Initialization method used to initialize weights.

    n_clusters: `int`, optional. Default:`None`, if we specify it , the algorithm will use K-means initialize the cluster center after autoencoder model trained.
    louvain_resolution: `list  or str or float. for example, louvain_resolution=1.2 or louvain_resolution=[0.2,0.4,0.8] or louvain_resolution="0.3,0.4,0.8" sep with ","
    n_neighbors, `int`, optional. Default: 10. The size of local neighborhood (in terms of number of neighboring data points) used for connectivity matrix. Larger values result in more global views of the manifold, while smaller values result in more local data being preserved. In general values should be in the range 2 to 100. Lo 

    pretrain_epochs:'int',optional. Default:`300`,the number of epochs for autoencoder model. 

    batch_size: `int`, optional. Default:`256`, the batch size for autoencoder model and clustering model. 

    activation; `str`, optional. Default,`relu`. the activation function for autoencoder model,which can be 'elu,selu,softplus,tanh,siogmid et al.', for detail please refer to`https://keras.io/activations/`.

    actincenter: `str`, optional. Default,'tanh', the activation function for the last layer in encoder and decoder model.

    drop_rate_SAE: `float`, optional. Default, `0.2`. The drop rate for Stacked autoencoder, which just for  finetuning. 

    is_stacked:`bool`,optional. Default,`True`.The model wiil be pretrained by stacked autoencoder if is_stacked==True.

    use_earlyStop:`bool`,optional. Default,`True`. Stops training if loss does not improve if given min_delta=1e-4, patience=10.

    use_ae_weights: `bool`, optional. Default, `True`. Whether use ae_weights that has been pretrained(which must saved in `save_dir/ae_weights.h5`)

    save_encoder_weights: `bool`, optional. Default, `False`, it will save inter_ae_weights for every 20 iterations. )

    save_dir: 'str',optional. Default,'result_tmp',some result will be saved in this directory.

    max_iter: `int`, optional. Default,`1000`. The maximum iteration for clustering.

    epochs_fit: `int or fload`,optional. Default,`4`, updateing clustering probability for each epochs_fit*n_sample, where n_sample is the sample size 

    num_Cores: `int`, optional. Default,`20`. How many cpus use during tranning. if `num_Cores` > the max cpus in our computer, num_Cores will use  a half of cpus in your computer. 

    use_GPU=True, `bool`, optional. Default, `True`. it will use GPU to train model if GPU is avaliable 

    random_seed, `int`,optional. Default,`201809`. the random seed for random.seed, numpy.random.seed, tensorflow.set_random_seed

    verbose,`bool`, optional. Default, `True`. It will ouput the model summary if verbose==True.

    do_tsne,`bool`,optional. Default, `False`. Whethter do tsne for representation or not.

    learning_rate,`float`,optional, Default(150).Note that the R-package "Rtsne" uses a default of 200. The learning rate can be a critical parameter. It should be between 100 and 1000. If the cost function increases during initial optimization, the early exaggeration factor or the learning rate might be too high. If the cost function gets stuck in a bad local minimum increasing the learning rate helps sometimes.

    perplexity, `float`, optional, Default(30). The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. The choice is not extremely critical since t-SNE is quite insensitive to this parameter.
    do_umap, `bool`, optional. Default, `False`,Whethter do umap for representation or not
    ------------------------------------------------------------------
    """

    if isinstance(data,AnnData):
        adata=data
    elif isinstance(data,pd.DataFrame):
        adata=sc.AnnData(data,obs=data.index,var=data.columns)
    else:
        x=data.toarray() if issparse(data) else data
        adata=sc.AnnData(x)

    if dims is None:
        dims=getdims(adata.shape)

    if isinstance(louvain_resolution,float) or isinstance(louvain_resolution,int):
        louvain_resolution=[float(louvain_resolution)]
    elif  isinstance(louvain_resolution,str):
        louvain_resolution=list(map(float,louvain_resolution.split(",")))
    else:
        assert isinstance(louvain_resolution,list),'louvain_resolution must be either a string with spearator "," or a list like [1.0,2.0,3.0] '
        louvain_resolution=list(map(float,louvain_resolution))
    #
    time_start=get_time()
    for ith,resolution in enumerate(louvain_resolution):
        print("Start to process resolution=",str(resolution))
        use_ae_weights=use_ae_weights if ith==0 else True
        res=train_single(data=data,
            dims=dims,
            alpha=alpha,
            tol=tol,
            init=init,
            n_clusters=n_clusters,
            louvain_resolution=resolution,
            n_neighbors=n_neighbors,
            pretrain_epochs=pretrain_epochs,
            epochs_fit=epochs_fit,
            batch_size=batch_size,
            activation=activation,
            actincenter=actincenter,
            drop_rate_SAE=drop_rate_SAE,
            is_stacked=is_stacked,
            use_earlyStop=use_earlyStop,
            use_ae_weights=use_ae_weights,
        save_encoder_weights=save_encoder_weights,
            save_encoder_step=save_encoder_step,
            save_dir=save_dir,
            max_iter=max_iter,
            num_Cores=num_Cores,
            num_Cores_tsne=num_Cores_tsne,
            use_GPU=use_GPU,
            random_seed=random_seed,
            verbose=verbose,
        do_tsne=do_tsne,
        learning_rate=learning_rate,
        perplexity=perplexity,
            do_umap=do_umap)
        #update adata
        data=res
    print("The run time for all resolution is:",get_time()-time_start)
    print("After training, the information of adata is:\n",adata)
    return data



class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=kb.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (kb.sum(kb.square(kb.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = kb.transpose(kb.transpose(q) / kb.sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DescModel(object):
    def __init__(self,
                 dims,
                 x, # input matrix, row sample, col predictors 
                 alpha=1.0,
         tol=0.005,
                 init='glorot_uniform', #initialization method
                 n_clusters=None,     # Number of Clusters, if provided, the clusters center will be initialized by K-means,
                 louvain_resolution=1.0, # resolution for louvain 
                 n_neighbors=10,    # the 
                 pretrain_epochs=300, # epoch for autoencoder
                 epochs_fit=4, #epochs for each update,int or float 
                 batch_size=256, #batch_size for autoencoder
                 random_seed=201809,
         activation='relu',
                 actincenter="tanh",# activation for the last layer in encoder, and first layer in the decoder 
                 drop_rate_SAE=0.2,
                 is_stacked=True,
                 use_earlyStop=True,
                 use_ae_weights=False,
         save_encoder_weights=False,
                 save_encoder_step=5,
                 save_dir="result_tmp"
                 # save result to save_dir, the default is "result_tmp". if recurvie path, the root dir must be exists, or there will be something wrong: for example : "/result_singlecell/dataset1" will return wrong if "result_singlecell" not exist
                 ):

        if not os.path.exists(save_dir):
            print("Create the directory:"+str(save_dir)+" to save result")
            os.mkdir(save_dir)
        self.dims = dims
        self.x=x #feature n*p, n:number of cells, p: number of genes
        self.alpha = alpha
        self.tol=tol
        self.init=init
        self.input_dim = dims[0]  # for clustering layer 
        self.n_stacks = len(self.dims) - 1
        self.is_stacked=is_stacked
        self.resolution=louvain_resolution
        self.n_neighbors=n_neighbors
        self.pretrain_epochs=pretrain_epochs
        self.epochs_fit=epochs_fit
        self.batch_size=batch_size
        self.random_seed=random_seed
        self.activation=activation
        self.actincenter=actincenter
        self.drop_rate_SAE=drop_rate_SAE
        self.is_stacked=is_stacked
        self.use_earlyStop=use_earlyStop
        self.use_ae_weights=use_ae_weights
        self.save_encoder_weights=save_encoder_weights
        self.save_encoder_step=save_encoder_step
        self.save_dir=save_dir
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
    #pretrain autoencoder
        self.pretrain(n_clusters=n_clusters)
        

    def pretrain(self,n_clusters=None):
        sae=SAE(dims=self.dims,
        act=self.activation,
                drop_rate=self.drop_rate_SAE,
                batch_size=self.batch_size,
                random_seed=self.random_seed,
                actincenter=self.actincenter,
                init=self.init,
                use_earlyStop=self.use_earlyStop,
                save_dir=self.save_dir
           )
        # begin pretraining
        t0 = get_time()
        print("Checking whether %s  exists in the directory"%str(os.path.join(self.save_dir,'ae_weights,h5')))
        if self.use_ae_weights: 
            if not os.path.isfile(self.save_dir+"/ae_weights.h5"):
                if self.is_stacked:
                    sae.fit(self.x,epochs=self.pretrain_epochs)
                else:
                    sae.fit2(self.x,epochs=self.pretrain_epochs)
                self.autoencoder=sae.autoencoders
                self.encoder=sae.encoder
            else:
                sae.autoencoders.load_weights(os.path.join(self.save_dir,"ae_weights.h5"))
                self.autoencoder=sae.autoencoders
                self.encoder=sae.encoder
        else:
            if self.is_stacked:
                sae.fit(self.x,epochs=self.pretrain_epochs)
            else:
                sae.fit2(self.x,epochs=self.pretrain_epochs)
            self.autoencoder=sae.autoencoders
            self.encoder=sae.encoder
        
        print('Pretraining time is', get_time() - t0)
        #save ae results into disk
        if not os.path.isfile(os.path.join(self.save_dir,"ae_weights.h5")):
            self.autoencoder.save_weights(os.path.join(self.save_dir,'ae_weights.h5'))
            self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights.h5'))
            print('Pretrained weights are saved to %s /ae_weights.h5' % self.save_dir)
        #save autoencoder model
        self.autoencoder.save(os.path.join(self.save_dir,"autoencoder_model.h5"))
        #initialize cluster centers using louvain if n_clusters is not exist
        features=self.extract_features(self.x)
        features=np.asarray(features)
        if isinstance(n_clusters,int):
            print("...number of clusters have been specified, Initializing Cluster centroid  using K-Means")
            kmeans = KMeans(n_clusters=n_clusters, n_init=20)
            Y_pred_init = kmeans.fit_predict(features)
            self.init_pred= np.copy(Y_pred_init)
            self.n_clusters=n_clusters
            cluster_centers=kmeans.cluster_centers_
            self.init_centroid=[cluster_centers]
        else:
            print("...number of clusters is unknown, Initialize cluster centroid using louvain method")
            #can be replaced by other clustering methods
            #using louvain methods in scanpy
            adata0=sc.AnnData(features)
            if adata0.shape[0]>200000:
                np.random.seed(adata0.shape[0])#set  seed 
                adata0=adata0[np.random.choice(adata0.shape[0],200000,replace=False)] 
            sc.pp.neighbors(adata0, n_neighbors=self.n_neighbors)
            sc.tl.louvain(adata0,resolution=self.resolution)
            Y_pred_init=adata0.obs['louvain']
            self.init_pred=np.asarray(Y_pred_init,dtype=int)
            if np.unique(self.init_pred).shape[0]<=1:
                #avoid only a cluster
                #print(np.unique(self.init_pred))
                exit("Error: There is only a cluster detected. The resolution:"+str(self.resolution)+"is too small, please choose a larger resolution!!")
            features=pd.DataFrame(adata0.X,index=np.arange(0,adata0.shape[0]))
            Group=pd.Series(self.init_pred,index=np.arange(0,adata0.shape[0]),name="Group")
            Mergefeature=pd.concat([features,Group],axis=1)
            cluster_centers=np.asarray(Mergefeature.groupby("Group").mean())
            self.n_clusters=cluster_centers.shape[0]
            self.init_centroid=[cluster_centers]
        #create desc clustering layer
        clustering_layer = ClusteringLayer(self.n_clusters,weights=self.init_centroid,name='clustering')(self.encoder.output)
        self.model = Model(inputs=self.encoder.input, outputs=clustering_layer)
        

    
    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_weights(weights)

    def extract_features(self, x):
        return self.encoder.predict(x)

    def predict(self, x):  # predict cluster labels using the output of clustering layer
        q = self.model.predict(x, verbose=0)
        return q.argmax(1)

    @staticmethod
    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def compile(self, optimizer='sgd', loss='kld'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit_on_batch(self,maxiter=1e4,update_interval=200,save_encoder_step=4):
        save_dir=self.save_dir
        #step1 initial weights by louvain,or Kmeans
        self.model.get_layer(name='clustering').set_weights(self.init_centroid)
        # Step 2: deep clustering
        y_pred_last = np.copy(self.init_pred)
        index_array = np.arange(self.x.shape[0])
        index=0
        for ite in range(int(maxiter)):
            if self.save_encoder_weights and ite%(save_encoder_step*update_interval)==0:
                self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights_resolution_'+str(self.resolution)+"_"+str(ite)+'.h5'))
                print('Fine tuning encoder weights are saved to %s/encoder_weights.h5' % self.save_dir) 
            if ite % update_interval ==0:
                q=self.model.predict(self.x,verbose=0)
                p=self.target_distribution(q)
                y_pred=q.argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                print("The value of delta_label of current",str(ite+1),"th iteration is",delta_label,">= tol",self.tol)
                if ite > 0 and delta_label < self.tol:
                    print('delta_label ', delta_label, '< tol ', self.tol)
                    print('Reached tolerance threshold. Stop training.')
                    break
            idx=index_array[index * self.batch_size: min((index+1) * self.batch_size, self.x.shape[0])]
            loss = self.model.train_on_batch(x=self.x[idx], y=p[idx])
            index = index + 1 if (index + 1) * self.batch_size <= self.x.shape[0] else 0
        #save encoder model
        self.encoder.save(os.path.join(self.save_dir,"encoder_model.h5"))
        #load model
        #encoder=load_model("encoder.h5")
        #
        y0=pd.Series(y_pred,dtype='category')
        y0.cat.categories=range(0,len(y0.cat.categories))
        print("The final prediction cluster is:")
        x=y0.value_counts()
        print(x.sort_index(ascending=True))
        Embedded_z=self.extract_features(self.x)
        q=self.model.predict(self.x,verbose=0)
        return Embedded_z,q

             
    
    def fit_on_all(self, maxiter=1e3, epochs_fit=5,save_encoder_step=5): # unsupervised
        save_dir=self.save_dir
        #step1 initial weights by louvain,or Kmeans
        self.model.get_layer(name='clustering').set_weights(self.init_centroid)
        # Step 2: deep clustering
        y_pred_last = np.copy(self.init_pred)
        for ite in range(int(maxiter)):
            if self.save_encoder_weights and ite%save_encoder_step==0: #save ae_weights for every 5 iterations
                self.encoder.save_weights(os.path.join(self.save_dir,'encoder_weights_resolution_'+str(self.resolution)+"_"+str(ite)+'.h5'))
                print('Fine tuning encoder weights are saved to %s/encoder_weights.h5' % self.save_dir)
            q = self.model.predict(self.x, verbose=0)
            p = self.target_distribution(q)  # update the auxiliary target distribution p
            # evaluate the clustering performance
            y_pred = q.argmax(1)
             # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0 and delta_label < self.tol:
                print('delta_label ', delta_label, '< tol ', self.tol)
                print('Reached tolerance threshold. Stop training.')
                break
            print("The value of delta_label of current",str(ite+1),"th iteration is",delta_label,">= tol",self.tol)
            #train on whole dataset on prespecified batch_size
            if self.use_earlyStop:
                callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=5,verbose=1,mode='auto')]
                self.model.fit(x=self.x,y=p,epochs=epochs_fit,batch_size=self.batch_size,callbacks=callbacks,shuffle=True,verbose=True)
            else:
                self.model.fit(x=self.x,y=p,epochs=epochs_fit,batch_size=self.batch_size,shuffle=True,verbose=True)
        #save encoder model
        self.encoder.save(os.path.join(self.save_dir,"encoder_model.h5"))
        #load model
        #encoder=load_model("encoder.h5")
        #
        
        y0=pd.Series(y_pred,dtype='category')
        y0.cat.categories=range(0,len(y0.cat.categories))
        print("The final prediction cluster is:")
        x=y0.value_counts()
        print(x.sort_index(ascending=True))
        Embedded_z=self.extract_features(self.x)
        return Embedded_z,q

    def fit(self,maxiter=1e4):
        if isinstance(self.epochs_fit,int):
            embedded_z,q=self.fit_on_all(maxiter=maxiter,epochs_fit=self.epochs_fit,save_encoder_step=self.save_encoder_step)
        else:
            import math
            update_interval=math.ceil(self.epochs_fit*self.x.shape[0]/self.batch_size)
            embedded_z,q=self.fit_on_batch(maxiter=maxiter,save_encoder_step=self.save_encoder_step,update_interval=update_interval)
        return embedded_z,q
         


class SAE(object):
    """ 
    Stacked autoencoders. It can be trained in layer-wise manner followed by end-to-end fine-tuning.
    For a 5-layer (including input layer) example:
        Autoendoers model: Input -> encoder_0->act -> encoder_1 -> decoder_1->act -> decoder_0;
        stack_0 model: Input->dropout -> encoder_0->act->dropout -> decoder_0;
        stack_1 model: encoder_0->act->dropout -> encoder_1->dropout -> decoder_1->act;
    
    Usage:
        from SAE import SAE
        sae = SAE(dims=[784, 500, 10])  # define a SAE with 5 layers
        sae.fit(x, epochs=100)
        features = sae.extract_feature(x)
        
    Arguments:
        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
              The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
        act: activation (default='relu'), not applied to Input, Hidden and Output layers.
        drop_rate: drop ratio of Dropout for constructing denoising autoencoder 'stack_i' during layer-wise pretraining
        batch_size: batch size
    """
    def __init__(self, dims, act='relu', drop_rate=0.2, batch_size=32,random_seed=201809,actincenter="tanh",init="glorot_uniform",use_earlyStop=True,save_dir='result_tmp'): #act relu
        self.dims = dims
        self.n_stacks = len(dims) - 1
        self.n_layers = 2*self.n_stacks  # exclude input layer
        self.activation = act
        self.actincenter=actincenter #linear
        self.drop_rate = drop_rate
        self.init=init
        self.batch_size = batch_size
        #set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        #
        self.use_earlyStop=use_earlyStop
        self.stacks = [self.make_stack(i) for i in range(self.n_stacks)]
        self.autoencoders ,self.encoder= self.make_autoencoders()
        #plot_model(self.autoencoders, show_shapes=True, to_file='autoencoders.png')

    def make_autoencoders(self):
        """ Fully connected autoencoders model, symmetric.
        """
        # input
        x = Input(shape=(self.dims[0],), name='input')
        h = x

        # internal layers in encoder
        for i in range(self.n_stacks-1):
            h = Dense(self.dims[i + 1], kernel_initializer=self.init,activation=self.activation, name='encoder_%d' % i)(h)

        # hidden layer,default activation is linear
        h = Dense(self.dims[-1],kernel_initializer=self.init, name='encoder_%d' % (self.n_stacks - 1),activation=self.actincenter)(h)  # features are extracted from here

        y=h
        # internal layers in decoder       
        for i in range(self.n_stacks-1, 0, -1):
            y = Dense(self.dims[i], kernel_initializer=self.init,activation=self.activation, name='decoder_%d' % i)(y)

        # output
        y = Dense(self.dims[0], kernel_initializer=self.init,name='decoder_0',activation=self.actincenter)(y)

        return Model(inputs=x, outputs=y,name="AE"),Model(inputs=x,outputs=h,name="encoder")

    def make_stack(self, ith):
        """ 
        Make the ith denoising autoencoder for layer-wise pretraining. It has single hidden layer. The input data is 
        corrupted by Dropout(drop_rate)
        
        Arguments:
            ith: int, in [0, self.n_stacks)
        """
        in_out_dim = self.dims[ith]
        hidden_dim = self.dims[ith+1]
        output_act = self.activation
        hidden_act = self.activation
        if ith == 0:
            output_act = self.actincenter# tanh, or linear
        if ith == self.n_stacks-1:
            hidden_act = self.actincenter #tanh, or linear
        model = Sequential()
        model.add(Dropout(self.drop_rate, input_shape=(in_out_dim,)))
        model.add(Dense(units=hidden_dim, activation=hidden_act, name='encoder_%d' % ith))
        model.add(Dropout(self.drop_rate))
        model.add(Dense(units=in_out_dim, activation=output_act, name='decoder_%d' % ith))
        return model

    def pretrain_stacks(self, x, epochs=200,decaying_step=3):
        """ 
        Layer-wise pretraining. Each stack is trained for 'epochs' epochs using SGD with learning rate decaying 10
        times every 'epochs/3' epochs.
        
        Arguments:
            x: input data, shape=(n_samples, n_dims)
            epochs: epochs for each stack
        """
        features = x
        for i in range(self.n_stacks):
            print( 'Pretraining the %dth layer...' % (i+1))
            for j in range(int(decaying_step)):  # learning rate multiplies 0.1 every 'epochs/4' epochs
                print ('learning rate =', pow(10, -1-j))
                self.stacks[i].compile(optimizer=SGD(pow(10, -1-j), momentum=0.9), loss='mse')
                if self.use_earlyStop is True:
                    callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
                    self.stacks[i].fit(features,features,callbacks=callbacks,batch_size=self.batch_size,epochs=math.ceil(epochs/decaying_step))
                else:
                    self.stacks[i].fit(x=features,y=features,batch_size=self.batch_size,epochs=math.ceil(epochs/decaying_step))
            print ('The %dth layer has been pretrained.' % (i+1))

            # update features to the inputs of the next layer
            feature_model = Model(inputs=self.stacks[i].input, outputs=self.stacks[i].get_layer('encoder_%d'%i).output)
            features = feature_model.predict(features)

    def pretrain_autoencoders(self, x, epochs=300):
        """
        Fine tune autoendoers end-to-end after layer-wise pretraining using 'pretrain_stacks()'
        Use SGD with learning rate = 0.1, decayed 10 times every 80 epochs
        
        :param x: input data, shape=(n_samples, n_dims)
        :param epochs: training epochs
        """
        print ('Copying layer-wise pretrained weights to deep autoencoders')
        for i in range(self.n_stacks):
            name = 'encoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())
            name = 'decoder_%d' % i
            self.autoencoders.get_layer(name).set_weights(self.stacks[i].get_layer(name).get_weights())

        print ('Fine-tuning autoencoder end-to-end')
        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -j)
            print ('learning rate =', lr)
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50)

    def fit(self, x, epochs=300,decaying_step=3):
        self.pretrain_stacks(x, epochs=int(epochs/2),decaying_step=decaying_step)
        self.pretrain_autoencoders(x, epochs=epochs)

    def fit2(self,x,epochs=300): #no stack directly traning 
        for j in range(math.ceil(epochs/50)):
            lr = pow(10, -j)
            print ('learning rate =', lr)
            self.autoencoders.compile(optimizer=SGD(lr, momentum=0.9), loss='mse')
            if self.use_earlyStop:
                callbacks=[EarlyStopping(monitor='loss',min_delta=1e-4,patience=10,verbose=1,mode='auto')]
                self.autoencoders.fit(x=x,y=x,callbacks=callbacks,batch_size=self.batch_size,epochs=epochs)
            else:
                self.autoencoders.fit(x=x, y=x, batch_size=self.batch_size, epochs=50)

    def extract_feature(self, x):
        """
        Extract features from the middle layer of autoencoders.
        
        :param x: data
        :return: features
        """
        hidden_layer = self.autoencoders.get_layer(name='encoder_%d' % (self.n_stacks - 1))
        feature_model = Model(self.autoencoders.input, hidden_layer.output)
        return feature_model.predict(x, batch_size=self.batch_size)
