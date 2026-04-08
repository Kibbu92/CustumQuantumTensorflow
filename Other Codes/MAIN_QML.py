import jax.numpy as jnp
import haiku as hk
import optax
import jax
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import gen_batches
from sklearn.utils import shuffle
from keras.datasets import mnist, fashion_mnist
from QCNN_QML import QCNN
    
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
#==============================================================================
#==============================================================================
def forward_fun(x):     
    x = QCNN(knl=KERNEL_SIZE, nL=NUM_LAYERS)(x)    
    x = x.reshape((len(x), -1))
    x = hk.Linear(NUM_CLASSES, name="full")(x) 
    x = jax.nn.softmax(x)
    return x
#==============================================================================
#==============================================================================
@jax.jit
def lossFn(params, x, y):
    yp = forward.apply(params,rng_key, x)    
    yp = jnp.clip(yp, 1e-16, 1.0)
    y = jax.nn.one_hot(y,NUM_CLASSES)
    loss = -jnp.sum(y * jnp.log(yp), axis=-1)
    return jnp.mean(loss)
#------------------------------------------------------------------------------
@jax.jit
def evaluate(params: hk.Params, images, labels) -> jax.Array:
    logits = forward.apply(params,rng_key, images)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.mean(predictions == labels)
#------------------------------------------------------------------------------
@jax.jit
def update(opt_state, params, images, labels):
    loss, grads = jax.value_and_grad(lossFn)(params, images, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, grads
###############################################################################
#################################### MAIN #####################################
###############################################################################
if __name__ == '__main__':
    
    LEARNING_RATE = 1e-3
    NUM_CLASSES   = 10 
    BATCH_SIZE    = 100
    SHRINK_FACTOR = 10
    EPOCHS        = 10     
    NUM_LAYERS    = 1          
    KERNEL_SIZE   = (2, 2)     
          
    ###########################################################################
    # Load Dataset (MNIST or FashionMNIST)
    dataset = Loader('FMNIST') 

    (X_train, Y_train), (X_test, Y_test) = dataset.load_data() 
    X_train = jnp.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in X_train])
    X_test  = jnp.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in X_test])
    X_train , X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.3, random_state=42)  
    
    X_train = X_train.astype('float32')/(2**8-1)
    X_test  = X_test.astype('float32')/(2**8-1)
    X_valid = X_valid.astype('float32')/(2**8-1)
    
    X_trn, Y_trn = X_train[::SHRINK_FACTOR], Y_train[::SHRINK_FACTOR]
    X_val, Y_val = X_valid[::SHRINK_FACTOR], Y_valid[::SHRINK_FACTOR]     
     
    ########################################################################### 
    rng_key = jax.random.PRNGKey(42)
    optimizer = optax.adam(LEARNING_RATE)              

    forward = hk.transform(forward_fun)        
    params = forward.init(rng=rng_key, x=X_trn[:BATCH_SIZE])                                              
    opt_state = optimizer.init(params)                
    #===========================================================================
    # Optimization Loop
    loss_trajectory = []
    param_trajectory = []
    grad_trajectory = []
    acc_train_trajectory = []
    acc_valid_trajectory = []

    for i in range(EPOCHS):
        
        X_trn, Y_trn = shuffle(X_trn, Y_trn, random_state=i)
        batch_slices = gen_batches(len(X_trn), BATCH_SIZE)
        
        if i==0:
            loss_value, grads = jax.value_and_grad(lossFn)(params, X_trn[:BATCH_SIZE], Y_trn[:BATCH_SIZE])
            
        acc_train = evaluate(params, X_trn, Y_trn)
        acc_valid = evaluate(params, X_val, Y_val)

        param_trajectory.append(params)
        loss_trajectory.append(loss_value)
        grad_trajectory.append(grads)
        acc_train_trajectory.append(acc_train)
        acc_valid_trajectory.append(acc_valid)
        
        print(f'step {i}, loss: {loss_value}, ACC-train: {acc_train}, ACC-test: {acc_valid}') 

        for batch in batch_slices:
            params, opt_state, loss_value, grads = update(opt_state, 
                                                          params, 
                                                          X_trn[batch], Y_trn[batch])      
    
    
    logits = forward.apply(params,rng_key, X_test[::SHRINK_FACTOR])
    predictions = jnp.argmax(logits, axis=-1)
    acc = jnp.mean(Y_test[::SHRINK_FACTOR]== predictions)      
    print(acc)
    
 
    plt.plot(acc_train_trajectory)
    plt.plot(acc_valid_trajectory)
    plt.show()
 