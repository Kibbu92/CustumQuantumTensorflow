import pennylane as qml
import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax

from typing import Optional, Callable

from flax import linen as nn
from sklearn.utils import gen_batches
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist

rng = jax.random.PRNGKey(42)
np.random.seed(2)

###############################################################################
###############################################################################
###############################################################################
def make_circuit(dev, n_qubits, n_layers):
    
    @qml.qnode(dev)
    def circuit(x, circuit_weights):    

        for n in range(n_qubits):
            qml.RZ(x[:,n],n)
            qml.RY(x[:,n],n)
            qml.RZ(x[:,n],n) 
            
        qml.templates.StronglyEntanglingLayers(circuit_weights, wires=range(n_qubits)) 
        
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    return jax.vmap(circuit,in_axes=(0,None))
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
class QuantumCircuit(nn.Module):
    
    num_qubits: int
    num_layers: int
    circuit: Callable

    @nn.compact
    def __call__(self, x):
        circuit_weights = self.param('circuit_weights',
                                     nn.initializers.normal(),
                                     (self.num_layers, self.num_qubits,3))
        x = self.circuit(x, circuit_weights)
        x = jnp.transpose(jnp.array(x),(1,2,0))
        return x
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
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
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
class QNN(nn.Module):
    
    circuit: Callable
    num_qubits: int
    num_layers: int
    num_labels: int

    def Layer(self, x, nf=32):
        x = nn.Conv(features=nf, kernel_size = (2,2),strides=(1, 1))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape = (3,3), strides= (1,1))        
        return x
    
    @nn.compact
    def __call__(self, x):
        
        x = self.Layer(x,4*self.num_qubits)
        x = self.Layer(x,2*self.num_qubits)
        x = self.Layer(x,  self.num_qubits)
        x = x.reshape((x.shape[0], -1, self.num_qubits)) 
        x = QuantumCircuit(num_qubits = self.num_qubits,
                           num_layers = self.num_layers,
                           circuit    = self.circuit,)(x)        
        x = x.reshape((x.shape[0], -1)) 
        x = nn.Dense(features=self.num_labels)(x)
        x = nn.softmax(x)
        return x     
############################################################################### 
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
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::  
@jax.jit
def CrossEntropyLoss(params, x, y_true):
    
    # y_pred = model.apply(params,rng,x)
    y_pred = model.apply(params, x)
    eps = 1e-10
    y_pred = jnp.clip(y_pred, a_min=eps, a_max=1.0 - eps)
    y_true = jax.nn.one_hot(y_true, num_classes=nC)    
    loss  = - jnp.sum(y_true * jnp.log(y_pred), axis=-1)
    return  jnp.mean(loss)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
@jax.jit
def evaluate(params, x, y_true):
    y_pred = model.apply(params, x)
    y_pred = jnp.argmax(y_pred, axis=-1)
    return jnp.mean(y_pred == y_true)    

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

    (X_train, Y_train), (X_test, Y_test) = dataset.load_data()
    X_train , X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)  

    _,w,h = X_train.shape  

    X_train = jnp.expand_dims(X_train, axis=-1).astype('float32')
    X_test  = jnp.expand_dims(X_test , axis=-1).astype('float32')
    X_valid = jnp.expand_dims(X_valid, axis=-1).astype('float32')

    ###########################################################################   
    nQ = 4    
    nL = 1 
    nC = 10
    
    dev = qml.device('default.qubit', wires=nQ)
    circuit = make_circuit(dev, nQ, nL)
    
    ###########################################################################
    LR = 1e-3
    Epoch = 10
    Batch = 100 
    
    ns = 10

    X_trn, Y_trn = X_train[::ns], Y_train[::ns]
    X_val, Y_val = X_valid[::ns], Y_valid[::ns]    
    
    ###########################################################################
    model = QNN(circuit=circuit,
                num_qubits=nQ,
                num_layers=nL,
                num_labels=nC)
    
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

        arg = (params, opt_state, X_trn, Y_trn,loss_val,acc)
        epoch_params = jax.lax.fori_loop(0, len(batches)-1, batch_loop, arg)    
        params, opt_state, data, targets, loss_val, acc = epoch_params    
        print(f'step {i}, loss: {loss_val}, ACC-train: {acc}') 

    
              
