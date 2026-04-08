import tensorflow as tf
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
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


def extract_patches_tensorflow(images, knl_size = (2,2), str_size=(1,1)):    
   
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1] + list(knl_size) + [1],  
        strides=[1] + list(str_size) + [1],  
        rates=[1, 1, 1, 1],  
        padding='SAME' 
    )

    return patches

###############################################################################
###############################################################################
###############################################################################
if __name__=='__main__':
    
    # GPU Initialitation 
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
    # Load Both Dataset (MNIST and FashionMNIST)
    (_, _), (X_test1, _) = mnist.load_data()
    (_, _), (X_test2, _) = fashion_mnist.load_data()

    X_test = np.concatenate((X_test1,X_test2),axis=0)
    np.random.shuffle(X_test)

    _,w,h = X_test.shape
    
    nL  = 1     # Numbres of Layers
    nQ  = 4     # Numbres of Qubits
    knl = ( int(np.log2(nQ)) , int(np.log2(nQ)) ) # Kernel size
    
    X_test  = np.expand_dims(X_test, axis=-1)
    X_patch = extract_patches_tensorflow(X_test, knl_size=knl).numpy()
    
    ###########################################################################   
    # Pennylane Circiut Initialization (as a benchmark)
    dev    = qml.device("default.qubit", wires=nQ)      
    Qcirc  = qml.QNode(quantum_circuit,dev,interface='tf') 
    ql_qml = qml.qnn.KerasLayer(Qcirc, weight_shapes={"weights": (nL, nQ, 3)}, output_dim=(w*h,nQ))  
    
    ###########################################################################   
    # Proposed Custum TF Quantum Circiut (to test)
    W = np.array(ql_qml.weights)[0]
    W = tf.convert_to_tensor(W, dtype=tf.float32)  
    ql_tf = QuantumLayer(nQ,nL=nL,weight=W)
    
    ###########################################################################   
    # Proposed Custum TF Quantum Circiut (to test)
    
    MEAN = []  
    
    N = 200      

    for j in range(len(X_patch)//N):        
        
        print(j)
       
        x = X_patch[j*N:(j+1)*N] 
        x = x.reshape(N,w*h,-1)

        Y_qml = ql_qml(x).numpy().reshape(N,w,h,-1)
        Y_tf  = ql_tf(x).numpy().reshape(N,w,h,-1)
    
        Y_qml = (Y_qml - np.min(Y_qml)) / (np.max(Y_qml) - np.min(Y_qml))
        Y_tf  = (Y_tf  - np.min(Y_tf) ) / (np.max(Y_tf)  - np.min(Y_tf ))        
      
        ind = np.random.randint(0,N,1)                        

        Mean = np.mean(Y_qml.astype('float32')-Y_tf, axis=(1,2,3))        
        MEAN.append(Mean)
        
        j = j + 1
     

    ###########################################################################
        
    MEAN = np.stack(MEAN).flatten()
    
    x = np.linspace(min(MEAN), max(MEAN), 100)
    
    mu_mean   = np.mean(MEAN)
    sig_mean  = np.std(MEAN)  
    
    MEAN_porb = norm.pdf(x, mu_mean, sig_mean)
    
    y_sig3 = norm.pdf(mu_mean+sig_mean, mu_mean, sig_mean)    

    sz = 11
    
    plt.figure(figsize=(5, 5))
    plt.hist(MEAN, color='green', bins=25, alpha=0.5, edgecolor='black', zorder=2, density=True)
    plt.plot(x, MEAN_porb, color = 'red',linewidth=2) 
    plt.plot([mu_mean,mu_mean], [0,MEAN_porb.max()], color = 'red',linewidth=1.5, linestyle='--') 
    plt.text(3*mu_mean, MEAN_porb.max(), '$\mu$ = %.2e' % (mu_mean), rotation = 0, fontsize = 10)
    plt.annotate("", (mu_mean-sig_mean, y_sig3), (mu_mean+sig_mean, y_sig3), arrowprops=dict(arrowstyle="<->", color='r', lw=1.5))
    plt.text(1.2*(mu_mean+sig_mean), y_sig3, '$\sigma$ = %.2e' % (sig_mean), rotation = 0, fontsize = 10)
    plt.grid(zorder=1)
    plt.xlabel('Output Error', fontsize=sz)
    plt.ylabel('Probability Density', fontsize=sz)
    plt.tick_params(axis='both', which='major', labelsize=sz)
    plt.title('Q-TF vs Pennylane on MNIST+FashionMNIST', fontsize=sz)    
    plt.xlim((min(MEAN), max(MEAN)))
    plt.tight_layout()
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
    ax.yaxis.get_offset_text().set_position((-0.07, 1.02)) 
    plt.show() 
    

    

  