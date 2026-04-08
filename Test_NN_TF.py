import tensorflow as tf
import pennylane as qml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
from keras.datasets import mnist, fashion_mnist
from QuLayer import QuantumLayer

tf.random.set_seed(42)
np.random.seed(2)
###############################################################################
###############################################################################
###############################################################################
def quantum_circuit(inputs : np.ndarray, weights : np.ndarray) -> list:  
    
    for n in range(nQ):
        qml.RZ(inputs[:,n],n)
        qml.RY(inputs[:,n],n)
        qml.RZ(inputs[:,n],n) 
    
    qml.templates.StronglyEntanglingLayers(weights, wires=range(nQ)) 
    
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(nQ)]
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def model(TF, KNL_Size):
    
    if TF:
        quantum_layer = ql_tf
    else:
        quantum_layer = ql_qml

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(w,h,1)),  
        tf.keras.layers.Conv2D(4*nQ, kernel_size = (2,2), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size = KNL_Size, strides=(1,1)),
        tf.keras.layers.Conv2D(2*nQ, kernel_size = (2,2), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size = KNL_Size, strides=(1,1)),
        tf.keras.layers.Conv2D(  nQ, kernel_size = (2,2), padding="same", activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size = KNL_Size, strides=(1,1)),
        tf.keras.layers.Reshape((-1, nQ)), 
        quantum_layer,  
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(nC, activation='softmax')
    ])
    
    return model
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
def Func(y_true, y_pred):  # Small epsilon for stability
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-10), axis=-1)  
    return tf.reduce_mean(loss)

###########################################################################
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
        
    (X_train, Y_train), (X_test, Y_test) = dataset.load_data()  # fashion_mnist
    X_train , X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)  

    _,w,h = X_train.shape  

    X_train = np.expand_dims(X_train, axis=-1)
    X_test  = np.expand_dims(X_test , axis=-1)
    X_valid = np.expand_dims(X_valid, axis=-1)

    ###########################################################################
    nQ = 4    
    nL = 1 
    nC = 10

    KNL_Size = (3,3)    
    
    dev    = qml.device("default.qubit", wires=nQ)      
    Qcirc  = qml.QNode(quantum_circuit,dev,interface='tf') 
    ql_qml = qml.qnn.KerasLayer(Qcirc, weight_shapes={"weights": (nL, nQ, 3)}, output_dim=((w-(KNL_Size[0]-1)*3)*(h-(KNL_Size[1]-1)*3),nQ))  
    ql_tf  = QuantumLayer(nQ, nL=nL, EmbType=1) 
    
    ###########################################################################
    LR = 1e-3
    Epoch = 10
    Batch = 100 
    
    ns = 10

    X_trn, Y_trn = X_train[::ns], Y_train[::ns]
    X_val, Y_val = X_valid[::ns], Y_valid[::ns]    
    
    ###########################################################################   
    train_dataset = tf.data.Dataset.from_tensor_slices((X_trn, np_utils.to_categorical(Y_trn)))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, np_utils.to_categorical(Y_val)))

    SHUFFLE_BUFFER_SIZE = X_trn.shape[0]
    train_batches = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(Batch)
    
    valid_batches = valid_dataset.batch(Batch)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
    TF = 1  # Set to 1 for TensorFlow (TF) layer, 0 for Pennylane layer   
    
    model = model(TF,KNL_Size=KNL_Size)
    model.compile(loss       = Func,
                  optimizer = Adam(learning_rate = LR), 
                  metrics   = ['accuracy'])  
    
    
    results_tf = model.fit(train_batches,
                           epochs=Epoch,
                           validation_data=valid_batches,
                           shuffle='batch')
    
    hist_df_tf = pd.DataFrame(results_tf.history) 
    hist_df_tf.plot( y=['loss','val_loss'])    
    hist_df_tf.plot( y=['accuracy','val_accuracy'])  
    


        
      