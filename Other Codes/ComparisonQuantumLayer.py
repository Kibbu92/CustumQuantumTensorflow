import pennylane as qml
import numpy as np
import jax 
import cv2
import tensorflow as tf
import itertools

from keras.datasets import mnist, fashion_mnist
pi = tf.constant(np.pi)     

class QCNN(tf.keras.Model):     
    def __init__(self, 
                 n_layer, 
                 seed = 42, 
                 kernel_size = (2,2), 
                 stride_size = (1,1),
                 w = None):
     
        super(QCNN, self).__init__() 
        
        self.kernel_size = kernel_size        
        self.stride_size = stride_size     
        self.n_layer     = n_layer
        self.n_qubit     = kernel_size[0]*kernel_size[1]
        self.seed        = seed

             
        self.wires = range(self.n_qubit)          
        
        CZ = tf.convert_to_tensor([[ 1, 0, 0, 0],
                                   [ 0, 1, 0, 0],
                                   [ 0, 0, 1, 0],
                                   [ 0, 0, 0,-1]],   dtype=tf.complex64 )
        
        self.CZ_m = tf.reshape(CZ, (2,2,2,2)) 
        self.w = w
         
            
    ###########################################################################
    def RotX(self, i, a):         
        cos_a = tf.cast(tf.math.cos(a / 2),tf.complex64)
        sin_a = tf.cast(tf.math.sin(a / 2),tf.complex64)
        j = tf.complex(0.0, 1.0)
        Rx = tf.convert_to_tensor([[     cos_a, -j * sin_a],
                                   [-j * sin_a,      cos_a]],   dtype=tf.complex64 )
        self.psi = tf.tensordot(Rx, self.psi, axes=([1], [i]))
        self.psi = tf.experimental.numpy.moveaxis(self.psi, 0, i)
        
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    def RotY(self, i, a):
        Ry = tf.convert_to_tensor([[tf.math.cos(a / 2), -tf.math.sin(a / 2)],
                                   [tf.math.sin(a / 2), tf.math.cos(a / 2)]], dtype=tf.complex64)
        self.psi = tf.tensordot(Ry, self.psi, axes=([1], [i]))
        self.psi = tf.experimental.numpy.moveaxis(self.psi, 0, i)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def RotZ(self, i, a):
        a = tf.cast(a, tf.complex64)
        Rz = tf.convert_to_tensor([[tf.math.exp(-1j * a / 2), 0],
                                   [0, tf.math.exp(1j * a / 2)]], dtype=tf.complex64)
        self.psi = tf.tensordot(Rz, self.psi, axes=([1], [i]))
        self.psi = tf.experimental.numpy.moveaxis(self.psi, 0, i)
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    def CZ(self, control, target):              
        self.psi = tf.tensordot(self.CZ_m, self.psi, ((2,3),(control, target))) 
        self.psi = tf.experimental.numpy.moveaxis(self.psi,(0,1),(control,target))   
 
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
    def Probs_TF(self):
        psi = tf.reshape(self.psi, (-1,))
        prob = tf.abs(psi) ** 2
        c1 = list(itertools.product([0, 1], repeat=self.n_qubit))
        c1 = tf.stack(c1)
        tmp = []

        for i in self.wires:
            tmp.append(c1[:, i])
        c1 = tf.stack(tmp)
        c1 = tf.transpose(c1)

        c2 = list(itertools.product([0, 1], repeat=len(self.wires)))
        c2 = tf.stack(c2)

        PROB = []
        for i in range(c2.shape[0]):
            ind = tf.reduce_all(tf.equal(c1, c2[i, :]), axis=1)
            PROB.append(tf.math.reduce_sum(prob[ind], axis=0))

        PROB = tf.stack(PROB)
        return PROB
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def PauliZ_ExpVal(self):
        psi = tf.reshape(self.psi, (-1,))
        prob = tf.abs(psi) ** 2

        c1 = list(itertools.product([0, 1], repeat=self.n_qubit))
        c1 = tf.stack(c1)
        Zval = []
        for i in self.wires:
            ind0 = tf.equal(c1[:, i], 0)
            ind1 = ~ind0
            Zval.append(tf.reduce_sum(prob[ind0]) - tf.reduce_sum(prob[ind1]))
        Zval = tf.stack(Zval)
        return Zval 
            
    def circuit(self,line): 
        
        for n in range(self.n_qubit):  
            self.RotZ(n, pi *line[n, 0])
            self.RotY(n, pi *line[n, 1])
            self.RotZ(n, pi *line[n, 2]) 
            
        for h in range(self.n_layer):            
            w_tmp = self.w[h]
            for i in range(self.n_qubit):        
                self.RotX(i, w_tmp[i, 0])
                self.RotY(i, w_tmp[i, 1])
                self.RotZ(i, w_tmp[i, 2]) 
                
            for ie in range(1, self.n_qubit, 2):
                self.CZ(ie-1 ,ie)  
                
            for ie in range(2, self.n_qubit, 2):
                self.CZ(ie-1 ,ie)  
                
    ###########################################################################
    def Map(self, x):
        def process_single_batch(batch):   
            def process_single_line(line): 
                
                line = tf.cast(line,tf.float32)
                
                psi = tf.zeros((2,) * self.n_qubit, dtype=tf.complex64)
                index = tf.constant((0,) * self.n_qubit)
                psi = tf.tensor_scatter_nd_update(psi, tf.reshape(index, (1, self.n_qubit)), [1.0])               
                self.psi = psi                     
                self.circuit(line)   
                
                return self.PauliZ_ExpVal()  
            
            return tf.vectorized_map(lambda single_line: process_single_line(single_line), batch)   
        return tf.vectorized_map(lambda single_batch: process_single_batch(single_batch), x) 
    
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


def mod_images(images):
    
    fun = lambda image, start_point: jax.lax.dynamic_slice(image, start_point, list(knl)+[3])     
    slice_im    = jax.vmap(fun, (None, 0))
    slice_im_ij = jax.vmap(slice_im, (0, None))    
    
    dim_i = images.shape[1] - knl[0]//2
    dim_j = images.shape[2] - knl[1]//2
    
    comb = np.array( list( itertools.product(np.arange(dim_i), np.arange(dim_j)) ) )
    O = np.zeros((comb.shape[0], 1), dtype=int)    
    comb = np.hstack((comb, O))
    result = slice_im_ij(images, comb)
    
    new_shape = list(result.shape)  # Ottieni la forma attuale
    new_shape[2] *= new_shape[3]  # Moltiplica le due dimensioni
    del new_shape[3]  # Rimuovi la seconda dimensione
    
    result = result.reshape(new_shape) 
    
    return result

def circuit(angles, data):
    
    for j in range(nQ):
        qml.Rot(np.pi * data[j, 0], 
                np.pi * data[j, 1], 
                np.pi * data[j, 2], 
                wires=j)

    for il in range(nL):
        # rotations
        for iq in range(nQ):
            qml.RX(angles[il, iq, 0], iq)
            qml.RY(angles[il, iq, 1], iq)
            qml.RZ(angles[il, iq, 2], iq)
        
        for ie in range(1, nQ, 2): # odd
            qml.CZ(wires=[ie-1, ie])
        
        for ie in range(2, nQ, 2): # even
            qml.CZ(wires=[ie-1, ie])       
   
    return [qml.expval(qml.PauliZ(i)) for i in range(nQ)]

def PreProcTF(inputs):
    patches = tf.image.extract_patches( images=inputs,
                                        sizes=[1] + list(knl) + [1],
                                        strides=[1 ,1, 1, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='VALID' )
    
    _,a,b,c = patches.shape
    
    patches = tf.reshape(patches,(-1,a*b,c))
    patches = tf.reshape(patches,(-1,*patches.shape[1:-1], nQ, 3))
    
    return patches 
###############################################################################
nL  = 1
knl = (2,2)
nQ  = knl[0]*knl[1]

dataset = Loader('FMNIST') 
(X_train, Y_train), (X_test, Y_test) = dataset.load_data() 
X_train = X_train[::100]

images  = np.array([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in X_train])
images2 = PreProcTF(images).numpy()
images3 = np.array(mod_images(images))

W = np.random.rand(nL,nQ,3)

dev   = qml.device("default.qubit", wires=range(nQ))
qnode = qml.QNode(circuit, dev)
circ  = jax.vmap(qnode, (None, 0)) 
circ  = jax.vmap(circ, (None, 0))

out = circ(W,images)
out = np.stack(out).transpose((1,2,0))

# ###############################################################################
qcnn = QCNN(nL,w=W,kernel_size=knl)
x = qcnn.Map(images)  
x = x.numpy()

d = (out-x)

print(d.min(),d.max())