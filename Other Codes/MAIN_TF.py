import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist, fashion_mnist
from QCNN_TF import QCNN

tf.random.set_seed(42)
np.random.seed(2)
###############################################################################
###############################################################################
###############################################################################
def model(quantum_layer): 
    model = tf.keras.Sequential([ quantum_layer,  
                                  tf.keras.layers.Flatten(),
                                  tf.keras.layers.Dense(nC, activation='softmax') ])
    return model

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def Func(y_true, y_pred):   
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-10), axis=-1)  
    return tf.reduce_mean(loss)

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
###############################################################################
###############################################################################
if __name__=='__main__':
    
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)  

    ###########################################################################
    # Load Dataset (MNIST or FashionMNIST)
    dataset = Loader('MNIST') 

    (X_train, Y_train), (X_test, Y_test) = dataset.load_data() 
    X_train = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in X_train])
    X_test  = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in X_test])
    X_train , X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)  
    
    X_train = X_train.astype('float32')/(2**8-1)
    X_test  = X_test.astype('float32')/(2**8-1)
    X_valid = X_valid.astype('float32')/(2**8-1)    
    
    ns  = 10
    X_trn, Y_trn = X_train[::ns], Y_train[::ns]
    X_val, Y_val = X_valid[::ns], Y_valid[::ns]     

    ###########################################################################     
    
    knl = (2, 2)  
    nL  = 4
    nC  = 10  
    ql_tf = QCNN(knl=knl, nL=nL) 
    
    ###########################################################################
    LR    = 1e-3
    Epoch = 10
    Batch = 100    
    
    ###########################################################################   
    train_dataset = tf.data.Dataset.from_tensor_slices((X_trn, np_utils.to_categorical(Y_trn)))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, np_utils.to_categorical(Y_val)))

    SHUFFLE_BUFFER_SIZE = X_trn.shape[0]
    train_batches = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(Batch)
    
    valid_batches = valid_dataset.batch(Batch)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  
    
    model = model(ql_tf)
    
    model.compile(loss = Func,
                  optimizer = Adam(learning_rate = LR), 
                  metrics = ['accuracy'])      
   
    results_tf = model.fit(train_batches,
                            epochs=Epoch,
                            validation_data=valid_batches,
                            shuffle='batch')
    
    hist_df_tf = pd.DataFrame(results_tf.history) 
    hist_df_tf.plot( y=['loss','val_loss'])    
    hist_df_tf.plot( y=['accuracy','val_accuracy'])  
    