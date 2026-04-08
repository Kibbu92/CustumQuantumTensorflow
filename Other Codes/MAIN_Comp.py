import cv2
import jax 
import optax
import haiku as hk
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import time 

from jax import numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
from sklearn.utils import shuffle
from keras.datasets import mnist, fashion_mnist
from tensorflow.keras.optimizers import Adam

from QCNN_QML import QCNN as QCNN_pj
from QCNN_TF  import QCNN as QCNN_tf

###############################################################################
################################# FUNCTIONS ###################################
###############################################################################        
def Loader(DataType):
    if DataType == 'FMNIST':
        dataset = fashion_mnist
    elif DataType == 'MNIST':        
        dataset = mnist
    else:
        dataset = mnist
        print('Warning: Unknown value for Data Type. Using default dataset (MNIST).')
    return dataset  
###############################################################################
@jax.jit
def lossFn(params, x, y_true):
    y_pred = Model_PJ.apply(params,rng_key, x)    
    loss = -jnp.sum(y_true * jnp.log(y_pred), axis=-1)
    return jnp.mean(loss)
#------------------------------------------------------------------------------
@jax.jit
def evaluate(params: hk.Params, images, labels) -> jax.Array:
    logits = Model_PJ.apply(params,rng_key, images)
    predictions = jnp.argmax(logits, axis=-1)
    labels = jnp.argmax(labels, axis=-1)
    return jnp.mean(predictions == labels)
#------------------------------------------------------------------------------
@jax.jit
def update(opt_state, params, images, labels):
    loss, grads = jax.value_and_grad(lossFn)(params, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, grads
#------------------------------------------------------------------------------
def model_PJ(x):
    x = QCNN_pj(knl=knl, nL=nL)(x) 
    x = x.reshape((len(x), -1))
    x = hk.Linear(nC, name="full")(x)
    x = jax.nn.softmax(x)
    return x
#==============================================================================
#==============================================================================
def Func(y_true, y_pred):  # Small epsilon for stability
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)  
    return tf.reduce_mean(loss)

def model_TF():  
    model = tf.keras.Sequential([ QCNN_tf(knl=knl, nL=nL) ,  
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(nC, activation='softmax') ])    
    return model
###############################################################################
###############################################################################
###############################################################################

if __name__ == '__main__':
    
    gpus = tf.config.list_physical_devices('GPU')    
    if gpus:
        try:
            logical_gpus = tf.config.list_logical_devices('GPU')    
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
            
    ###########################################################################
    ################ Load Dataset (MNIST or FashionMNIST) #####################
    ###########################################################################
    dataset = Loader('MNIST') 
   
    (X_train, Y_train), (X_test, Y_test) = dataset.load_data() 
    X_train = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in X_train])
    X_test  = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in X_test])
    X_train , X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)  
    
    X_train = X_train.astype('float32')/(2**8-1)
    X_test  = X_test.astype('float32')/(2**8-1)
    X_valid = X_valid.astype('float32')/(2**8-1)    
    
    ns = 10

    X_trn, Y_trn = X_train[::ns], Y_train[::ns]
    X_val, Y_val = X_valid[::ns], Y_valid[::ns]   
    
    X_trn, Y_trn = shuffle(X_trn, Y_trn, random_state=42)
    
    nC = 10 
    Y_trn = jax.nn.one_hot(Y_trn,nC)
    Y_val = jax.nn.one_hot(Y_val,nC)

    ###########################################################################
    ########### Models Initialization (Pennylane and Tensorflow) ##############
    ###########################################################################
    knl = (2, 2)  
    nL  = 2
    
    Learn  = 1e-3    
    Epochs = 10    
    Batch  = 100   
    
    ########################################################################### 
    # Pennylane/Jax Training
    rng_key = jax.random.PRNGKey(42)
    optimizer = optax.adam(Learn, eps=1e-7) 
    
    Model_PJ  = hk.transform(model_PJ)
    W0 = Weigths = Model_PJ.init(rng=rng_key, x=X_trn[:Batch]) 
    opt_state = optimizer.init(Weigths)                
    
    loss_trajectory      = []
    param_trajectory     = []
    grad_trajectory      = []
    acc_train_trajectory = []
    acc_valid_trajectory = []
    
    DT = []
    for i in range(Epochs):
        
        batch_slices = gen_batches(len(X_trn), Batch)
        
        if i==0:
            loss_value, grads = jax.value_and_grad(lossFn)(Weigths, X_trn[:Batch], Y_trn[:Batch])
            
        acc_train = evaluate(Weigths, X_trn, Y_trn)
        acc_valid = evaluate(Weigths, X_val, Y_val)

        param_trajectory.append(Weigths)
        loss_trajectory.append(loss_value)
        grad_trajectory.append(grads)
        acc_train_trajectory.append(acc_train)
        acc_valid_trajectory.append(acc_valid)
        
        print(f'step {i}, loss: {loss_value}, ACC-train: {acc_train}, ACC-test: {acc_valid}') 
        for batch in batch_slices:
            T0 = time.time()

            Weigths, opt_state, loss_value, grads = update(opt_state, 
                                                            Weigths, 
                                                            X_trn[batch], Y_trn[batch])      
            TF = time.time()
            DT.append( (TF-T0)/Batch)

    DT = np.mean(DT)
    
    T0 = time.time()    
    Yp_PJ = Model_PJ.apply(Weigths,rng_key,X_test[::2])
    TF = time.time()
    print((TF-T0)/len(X_test[::2]))
    
    Yp_PJ = jnp.argmax(Yp_PJ, axis=-1)
    acc   = jnp.mean(Y_test[::2]==Yp_PJ)      
    print(DT)
    print(acc) # 2.22 ms/image

    plt.plot(acc_train_trajectory)
    plt.plot(acc_valid_trajectory)
    plt.show()
    
    Wq = np.array(W0['qcnn']['angles'])
    Wf, Bf =  np.array(W0['full']['w']), np.array(W0['full']['b'])
    
    W = [Wq,Wf,Bf]

    ########################################################################### 
    # Tensorflow Training   
    train_dataset = tf.data.Dataset.from_tensor_slices((X_trn, Y_trn))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))

    train_batches = train_dataset.batch(Batch)    
    valid_batches = valid_dataset.batch(Batch)
    
    Model_TF = model_TF()
    Model_TF.compile(loss = Func,
                     optimizer = Adam(learning_rate = Learn), 
                     metrics = ['accuracy'])     
    Model_TF.build(X_trn[:1].shape)
    Model_TF.set_weights(W)   
    
    results_tf = Model_TF.fit(train_batches,
                              epochs=Epochs,
                              validation_data=valid_batches,
                              shuffle='False')
    
    hist_df_tf = pd.DataFrame(results_tf.history) 
    hist_df_tf.plot( y=['loss','val_loss'])    
    hist_df_tf.plot( y=['accuracy','val_accuracy'])  
    
    T0 = time.time()
    Yp_TF = Model_TF.predict(X_test[::2],batch_size=100)
    TF = time.time()
    print((TF-T0)/len(X_test[::2]))
    
    Yp_TF = np.argmax(Yp_TF, axis=-1)
    acc = np.mean(Y_test[::2]==Yp_TF)      
    print(acc) # 0.85 ms/images
    
    plt.plot(results_tf.history['loss'])
    plt.plot(loss_trajectory)
    
    plt.plot(results_tf.history['accuracy'])
    plt.plot(acc_train_trajectory)
 
    plt.plot(results_tf.history['val_accuracy'])
    plt.plot(acc_valid_trajectory)