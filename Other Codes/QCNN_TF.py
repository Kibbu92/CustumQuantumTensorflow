import tensorflow as tf
import numpy as np
import itertools
from typing import  Tuple

pi = tf.constant(np.pi)     
class QCNN(tf.keras.Model):     
    def __init__(self, 
                 knl: Tuple = (2, 2), 
                 nL: int = 4):
     
        super(QCNN, self).__init__() 
        
        self.nQ   = knl[0]*knl[1]
        self.nL   = nL
        self.knl  = knl             
        
        CZ = tf.convert_to_tensor([[ 1, 0, 0, 0],
                                   [ 0, 1, 0, 0],
                                   [ 0, 0, 1, 0],
                                   [ 0, 0, 0,-1]],   dtype=tf.complex64 )
        
        self.CZ_m = tf.reshape(CZ, (2,2,2,2)) 
           
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::         
    def build(self, input_shape):  
        
        ini = tf.keras.initializers.RandomUniform( minval=-pi, maxval=pi, seed=42)        
        self.w = self.add_weight(shape=(self.nL, self.nQ,3),
                                 initializer=ini,
                                 trainable=True)          
            
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
        c1 = list(itertools.product([0, 1], repeat=self.nQ))
        c1 = tf.stack(c1)
        tmp = []

        for i in range(self.nQ) :
            tmp.append(c1[:, i])
        c1 = tf.stack(tmp)
        c1 = tf.transpose(c1)

        c2 = list(itertools.product([0, 1], repeat=self.nQ))
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

        c1 = list(itertools.product([0, 1], repeat=self.nQ))
        c1 = tf.stack(c1)
        Zval = []
        for i in range(self.nQ) :
            ind0 = tf.equal(c1[:, i], 0)
            ind1 = ~ind0
            Zval.append(tf.reduce_sum(prob[ind0]) - tf.reduce_sum(prob[ind1]))
        Zval = tf.stack(Zval)
        return Zval 
    ###########################################################################       
    def circuit(self,line):         

        for n in range(self.nQ):  
            self.RotZ(n, pi *line[n, 0])
            self.RotY(n, pi *line[n, 1])
            self.RotZ(n, pi *line[n, 2]) 
            
        for h in range(self.nL):            
            w_tmp = self.w[h]
            for i in range(self.nQ):        
                self.RotX(i, w_tmp[i, 0])
                self.RotY(i, w_tmp[i, 1])
                self.RotZ(i, w_tmp[i, 2]) 
                
            for ie in range(1, self.nQ, 2):
                self.CZ(ie-1 ,ie)  
                
            for ie in range(2, self.nQ, 2):
                self.CZ(ie-1 ,ie)                  
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    def Map(self, x):
        def process_single_batch(batch):   
            def process_single_line(line): 
                
                line = tf.cast(line,tf.float32)

                psi = tf.zeros((2,) * self.nQ, dtype=tf.complex64)
                index = tf.constant((0,) * self.nQ)
                psi = tf.tensor_scatter_nd_update(psi, tf.reshape(index, (1, self.nQ)), [1.0])                
                self.psi = psi 
                    
                self.circuit(line)   
                
                return self.PauliZ_ExpVal()  
            
            return tf.vectorized_map(lambda single_line: process_single_line(single_line), batch)   
        return tf.vectorized_map(lambda single_batch: process_single_batch(single_batch), x) 
       
    ###########################################################################
    ###########################################################################
    def PreProc(self, images):
        patches = tf.image.extract_patches( images=images,
                                            sizes=[1] + list(self.knl) + [1],
                                            strides=[1, 1, 1, 1],
                                            rates=[1, 1, 1, 1],
                                            padding='VALID' )
        
        _,a,b,c = patches.shape        
        patches = tf.reshape(patches,(-1,a*b,c))
        patches = tf.reshape(patches,(-1,*patches.shape[1:-1], self.nQ, 3))
        
        return patches 
    
    @tf.function
    def call(self, x):
        x = self.PreProc(x) 
        x = self.Map(x)   
        return x