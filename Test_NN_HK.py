import pennylane as qml
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
import time

from typing import Optional
import jax.numpy as jnp
import haiku as hk
import jax
import optax

from sklearn.utils import gen_batches
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.datasets import mnist, fashion_mnist

rng = jax.random.PRNGKey(42)
np.random.seed(2)

###############################################################################
###############################################################################
###############################################################################   
class QLayer(hk.Module):
    
    def __init__(self, qcircuit, n_qubits, n_layers: int = 2, name: Optional[str] = 'QuantumLayer'):
        
        super().__init__(name=name)  

        dev = qml.device("default.qubit", wires=range(n_qubits))
        qnode = qml.QNode(qcircuit, dev)
        circ = jax.vmap(qnode, (0, None)) 
        
        self.circ = circ      
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
    def __call__(self, x):
        
        weight_init = hk.initializers.RandomUniform(-jnp.pi, jnp.pi)
        weight = hk.get_parameter("Qweight", shape=[self.n_layers, self.n_qubits,3], dtype=x.dtype, init=weight_init)

        result = self.circ(x,weight)
        result = jnp.stack(result)
        result = jnp.moveaxis(result, 0, -1)
        return result
    
###############################################################################
def quantum_circuit(inputs : np.ndarray, weights : np.ndarray) -> list:  
    
    for n in range(nQ):
        qml.RZ(inputs[:,n],n)
        qml.RY(inputs[:,n],n)
        qml.RZ(inputs[:,n],n) 
    
    qml.templates.StronglyEntanglingLayers(weights, wires=range(nQ)) 
    
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(nQ)]


class hqnn(hk.Module):
    def __init__(self, qcircuit):
        super().__init__(name="HQNN")
        
        
        self.conv1 = hk.Conv2D(output_channels=4*nQ, kernel_shape=(2,2), padding="SAME",name='Conv1')
        self.conv2 = hk.Conv2D(output_channels=2*nQ, kernel_shape=(2,2), padding="SAME",name='Conv2')
        self.conv3 = hk.Conv2D(output_channels=  nQ, kernel_shape=(2,2), padding="SAME",name='Conv3')
        
        self.MaxPool = hk.MaxPool(window_shape=(1,3,3,1), strides=(1,1,1,1), padding="VALID")        
        self.Reshape = hk.Reshape(output_shape=(-1,nQ))
        self.qcnn = QLayer(qcircuit, n_qubits=nQ, n_layers=nL)
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(nC,name='Dense')

    def __call__(self, x):
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.MaxPool(x)
        
        x = self.conv2(x)
        x = jax.nn.relu(x)
        x = self.MaxPool(x)
        
        x = self.conv3(x)
        x = jax.nn.relu(x)
        x = self.MaxPool(x)
        
        x = self.Reshape(x)
        x = self.qcnn(x)         
        x = self.flatten(x)
        
        x = self.linear(x)
        x = jax.nn.softmax(x)
        
        return x
    
    
def HQNN(x):    
    x = hqnn(quantum_circuit)(x)
    return x

@jax.jit
def CrossEntropyLoss(params, x, y_true):
    
    y_pred = model.apply(params,rng,x)
    
    eps = 1e-10
    y_pred = jnp.clip(y_pred, a_min=eps, a_max=1.0 - eps)
    y_true = jax.nn.one_hot(y_true, num_classes=nC)    
    loss  = - jnp.sum(y_true * jnp.log(y_pred), axis=-1)
    return  jnp.mean(loss)

@jax.jit
def evaluate(params, x, y_true):
    y_pred = model.apply(params,rng,x)    
    y_pred = jnp.argmax(y_pred, axis=-1)
    return jnp.mean(y_pred == y_true)

@jax.jit
def update(opt_state, params, x, y_true):
    loss, grads = jax.value_and_grad(CrossEntropyLoss)(params, x, y_true)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, grads


@jax.jit
def batch_loop(idx, arg_batch):
    params, opt_state, data, targets, loss_val, acc = arg_batch 
    x = jax.lax.dynamic_slice(data , (idx*Batch,0,0,0) ,(Batch, data.shape[1], data.shape[2], data.shape[3]))
    y = jax.lax.dynamic_slice(targets , (idx*Batch,) ,(Batch,))    
    loss_val, grads = jax.value_and_grad(CrossEntropyLoss)(params, x, y)     
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    acc = evaluate(params, x, y) 

    return (params, opt_state, data, targets, loss_val, acc)

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
    
    ###########################################################################
    # Load Dataset (MNIST or FashionMNIST)
    dataset = Loader('MNIST')  
        
    (X_train, Y_train), (X_test, Y_test) = dataset.load_data()  # fashion_mnist
    X_train , X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)  

    _,w,h = X_train.shape  

    X_train = np.expand_dims(X_train, axis=-1).astype('float32')
    X_test  = np.expand_dims(X_test , axis=-1).astype('float32')
    X_valid = np.expand_dims(X_valid, axis=-1).astype('float32')

    ###########################################################################
    nQ = 4    
    nL = 1 
    nC = 10    
    
    ###########################################################################
    LR = 1e-3
    Epoch = 10
    Batch = 100 
    
    ns = 10

    X_trn, Y_trn = X_train[::ns], Y_train[::ns]
    X_val, Y_val = X_valid[::ns], Y_valid[::ns]    
    
    ###########################################################################

    model = hk.transform(HQNN)
    rng = jax.random.PRNGKey(42)
    
    params = model.init(rng, X_trn[:3])
    optimizer = optax.adam(LR,eps=1e-7)
    opt_state = optimizer.init(params)     
    
    
    X_trn, Y_trn = shuffle(X_trn, Y_trn, random_state=42)
    batch_slices = gen_batches(len(Y_trn), Batch)
    
    loss_trajectory = []
    acc_train_trajectory = []
    acc_valid_trajectory = []
    
    batches = jnp.arange((X_trn.shape[0]//Batch)+1)
   
    for i in range(Epoch):
        
        if i == 0:
            loss_val, grads = jax.value_and_grad(CrossEntropyLoss)(params, X_trn[:Batch], Y_trn[:Batch]) 
            acc = evaluate(params, X_trn[:Batch], Y_trn[:Batch]) 

        T0 = time.perf_counter() 
        arg = (params, opt_state, X_trn, Y_trn,loss_val,acc)
        epoch_params = jax.lax.fori_loop(0, len(batches)-1, batch_loop, arg)    
        params, opt_state, data, targets, loss_val, acc = epoch_params          
        
        TF = time.perf_counter()        
        print((TF-T0)/len(X_trn))      

        print(f'step {i}, loss: {loss_val}, ACC-train: {acc}') 

    
              
