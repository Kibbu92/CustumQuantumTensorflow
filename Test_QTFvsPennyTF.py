import tensorflow as tf
import pennylane as qml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils

from keras.datasets import mnist, fashion_mnist

from QuLayer import QuantumLayer

tf.random.set_seed(22)
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

#------------------------------------------------------------------------------
def model(quantum_layer, KNL_Size=(3,3)):   
   
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
        tf.keras.layers.Dense(n_class, activation='softmax')
        
    ])
    
    return model

#------------------------------------------------------------------------------
def Func(y_true, y_pred):  
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-10), axis=-1)  # Add a small epsilon for stability
    return tf.reduce_mean(loss)

###############################################################################
def Loader(DataType):
    if DataType == 'FMNIST':
        dataset = fashion_mnist
    elif DataType == 'MNIST':        
        dataset = mnist
    else:
        dataset = mnist
        print(f"WANRING!: Unknown value '{DataType}' for Data Type. Using default dataset (MNIST).")
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
    # PARAMETERS TO SET 
    
    nQ = 4              # Number of Qubits
    nL = 1              # Number of Layers
    KNL_Size = (3,3)    # Kernel of MaxPooling  
        
    DataType = 'MNIST'  # Which dataset to test: 'MNIST' or 'FMINST'  
    ns       = 10       # Number of rows to skip to reduce the size of the dataset

    LR       = 1e-3     # Training Learning Rate
    Epochs   = 10       # Training Epochs
    Batch    = 100      # Training Batch
    
    
    ###########################################################################
    # Load Dataset (MNIST or FashionMNIST)
    
    dataset = Loader(DataType) 

    (X_train, Y_train), (X_test, Y_test) = dataset.load_data() 
    X_train , X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)  

    _,w,h = X_train.shape  

    X_train = np.expand_dims(X_train, axis=-1)
    X_test  = np.expand_dims(X_test , axis=-1)
    X_valid = np.expand_dims(X_valid, axis=-1)
   
    n_class = len(np.unique(Y_train))

    X_trn, Y_trn = X_train[::ns], Y_train[::ns]
    X_val, Y_val = X_valid[::ns], Y_valid[::ns]  
    
    ###########################################################################
    # QUANTUM LAYERS INITIALIZATION  
    
    # Pannylane/TF Layer
    dev    = qml.device("default.qubit", wires=nQ)      
    Qcirc  = qml.QNode(quantum_circuit,dev,interface='tf') 
    ql_qml = qml.qnn.KerasLayer(Qcirc, weight_shapes={"weights": (nL, nQ, 3)}, output_dim=((w-(KNL_Size[0]-1)*3)*(h-(KNL_Size[1]-1)*3),nQ))  
    
    # Proposed Q-TF Layer
    ql_tf  = QuantumLayer(nQ, nL=nL, EmbType=0) 
     
    ###########################################################################
    # TRAINING 
   
    train_dataset = tf.data.Dataset.from_tensor_slices((X_trn, np_utils.to_categorical(Y_trn)))
    valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, np_utils.to_categorical(Y_val)))

    SHUFFLE_BUFFER_SIZE = X_trn.shape[0]
    train_batches = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(Batch)
    
    valid_batches = valid_dataset.batch(Batch)
    
    #--------------------------------------------------------------------------
    # Pannylane/TF Layer
    model_qml = model(ql_qml, KNL_Size=KNL_Size)
    model_qml.compile(loss      = Func,
                      optimizer = Adam(learning_rate = LR), 
                      metrics   = ['accuracy']) 

    initial_weights = model_qml.get_weights()    

    results_qml = model_qml.fit(train_batches,
                                epochs=Epochs,
                                validation_data=valid_batches,
                                shuffle='batch')   

    hist_df_qml = pd.DataFrame(results_qml.history) 
     
    #--------------------------------------------------------------------------
    # Proposed Q-TF Layer

    model_tf = model(ql_tf, KNL_Size=KNL_Size)
    model_tf.compile(loss       = Func,
                      optimizer = Adam(learning_rate = LR), 
                      metrics   = ['accuracy'])  
    
    model_tf.set_weights(initial_weights) 
    
    results_tf = model_tf.fit(train_batches,
                              epochs=Epochs,
                              validation_data=valid_batches,
                              shuffle='batch')
    
    hist_df_tf = pd.DataFrame(results_tf.history) 
  
    
    ###########################################################################

    plt.figure(figsize=(10, 6))
    
    # Loss Pannylane/TF 
    plt.plot(hist_df_qml['loss'], label='QML - Training Loss', linestyle='-')
    plt.plot(hist_df_qml['val_loss'], label='QML - Validation Loss', linestyle='--')
    
    # Loss Q-TF 
    plt.plot(hist_df_tf['loss'], label='TF - Training Loss', linestyle='-')
    plt.plot(hist_df_tf['val_loss'], label='TF - Validation Loss', linestyle='--')
    
    plt.title('Confronto delle perdite (Loss)')
    plt.xlabel('Epoche')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    #--------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    # Accuracy Pannylane/TF
    plt.plot(hist_df_qml['accuracy'], label='QML - Training Accuracy', linestyle='-')
    plt.plot(hist_df_qml['val_accuracy'], label='QML - Validation Accuracy', linestyle='--')
    
    # Accuracy  Q-TF
    plt.plot(hist_df_tf['accuracy'], label='TF - Training Accuracy', linestyle='-')
    plt.plot(hist_df_tf['val_accuracy'], label='TF - Validation Accuracy', linestyle='--')
    
    plt.title('Confronto delle accuratezze (Accuracy)')
    plt.xlabel('Epoche')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    
        
      