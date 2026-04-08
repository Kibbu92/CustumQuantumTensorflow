"""
Created on Mon Dec  9 14:36:37 2025

Author: Andrea Carbone

"""

import tensorflow as tf
import itertools

class QuantumLayer(tf.keras.Model):
   
    ###########################################################################
    ############################# LAYER INITIALIZATION ########################
    ###########################################################################
    def __init__(self, nQ, nL=2, ini=tf.keras.initializers.GlorotUniform(42), 
                 Out_Wires=None, name='QuantumLayer', weight=None, EmbType=1, OutType=1):
        super(QuantumLayer, self).__init__(name=name)
        self.nQ = nQ
        self.nL = nL
        self.ini = ini
        self.EmbType = EmbType
        self.OutType = OutType
        
        den = tf.cast(tf.sqrt(1/2), tf.complex64)

        CNOT_matrix = tf.convert_to_tensor([[1,0,0,0],
                                            [0,1,0,0],
                                            [0,0,0,1],
                                            [0,0,1,0]], dtype=tf.complex64)

        self.CNOT_m = tf.reshape(CNOT_matrix, (2, 2, 2, 2))          
        self.H_m    = den*tf.convert_to_tensor([[ 1, 1,], [ 1,-1 ]], dtype=tf.complex64)        
        self.XGate  = tf.convert_to_tensor([[0, 1], [1, 0]], dtype=tf.complex64)  

        self.weight = weight

        if Out_Wires is None:
            self.wires = range(self.nQ)
        else:
            self.wires = Out_Wires
            
    ###########################################################################
    ########################### WEIGHTS INIZIALIZATION ########################
    ###########################################################################
    def build(self, input_shape):
        if self.weight is None:
            self.w = self.add_weight(shape=(self.nL, self.nQ, 3),
                                     initializer=self.ini,
                                     trainable=True,
                                     dtype=tf.float32)
        else:
            self.w = self.weight

    ###########################################################################
    ############################## BASIC OPERETIONS ###########################
    ###########################################################################
    def RotY(self, i, a):
        Ry = tf.convert_to_tensor([[tf.math.cos(a / 2), -tf.math.sin(a / 2)],
                                   [tf.math.sin(a / 2), tf.math.cos(a / 2)]], dtype=tf.complex64)
        self.psi = tf.tensordot(Ry, self.psi, axes=([1], [i]))
        self.psi = tf.experimental.numpy.moveaxis(self.psi, 0, i)

    def RotZ(self, i, a):
        a = tf.cast(a, tf.complex64)
        Rz = tf.convert_to_tensor([[tf.math.exp(-1j * a / 2), 0],
                                   [0, tf.math.exp(1j * a / 2)]], dtype=tf.complex64)
        self.psi = tf.tensordot(Rz, self.psi, axes=([1], [i]))
        self.psi = tf.experimental.numpy.moveaxis(self.psi, 0, i)
    
    def CNOT(self, control, target):
        self.psi = tf.tensordot(self.CNOT_m, self.psi, ((2, 3), (control, target)))
        self.psi = tf.experimental.numpy.moveaxis(self.psi, (0, 1), (control, target))
        
    def H(self, i):        
        self.psi = tf.tensordot(self.H_m, self.psi, axes=([1], [i]))
        self.psi = tf.experimental.numpy.moveaxis(self.psi, 0, i)
    
    # LIST TO DO
    #==========================================================================
    #... add other pre-made operations       
    #==========================================================================
    #... add possibility to have custum operations as input 
    #==========================================================================
    
    ###########################################################################
    ############################## QUANTUM CIRCUITS ###########################
    ###########################################################################
    def StrongEntagled(self):
        for h in range(self.nL):
            w_tmp = self.w[h]
            for i in range(self.nQ):                
                self.RotZ(i, w_tmp[i, 0])
                self.RotY(i, w_tmp[i, 1])
                self.RotZ(i, w_tmp[i, 2])

            for i in range(self.nQ):
                self.CNOT(i % self.nQ, (i + (h + 1)) % self.nQ)
    
    # LIST TO DO
    #==========================================================================
    #... add other pre-made circuits       
    #==========================================================================
    #... add possibility to have custum circuit as input 
    #==========================================================================
    
    ###########################################################################
    ############################## QUANTUM OUTPUT #############################
    ###########################################################################
    #==========================================================================
    # Probability Output 
    #==========================================================================
    def Probs_TF(self):
        psi = tf.reshape(self.psi, (-1,))
        prob = tf.abs(psi) ** 2
        c1 = list(itertools.product([0, 1], repeat=self.nQ))
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
    
    #==========================================================================
    # PauliZ Output 
    #==========================================================================
    def PauliZ_ExpVal(self):
        psi = tf.reshape(self.psi, (-1,))
        prob = tf.abs(psi) ** 2

        c1 = list(itertools.product([0, 1], repeat=self.nQ))
        c1 = tf.stack(c1)

        Zval = []
        for i in self.wires:
            ind0 = tf.equal(c1[:, i], 0)
            ind1 = ~ind0
            Zval.append(tf.reduce_sum(prob[ind0]) - tf.reduce_sum(prob[ind1]))
        Zval = tf.stack(Zval)

        return Zval
    
    ###########################################################################
    ########################### QUANTUM LAYER MAPPING #########################
    ###########################################################################
    def Map(self, x):
        
        def process_single_batch(batch):
            """
            Function to process a single batch (shape: (N, M)).
            """
            def process_single_line(line):
                
                #==========================================================
                # INPUT                 
                if self.EmbType == 1:
                    
                    #==========================================================
                    # ANLGES EMBEDDING 
                    #==========================================================
                    psi = tf.zeros((2,) * self.nQ, dtype=tf.complex64)
                    index = tf.constant((0,) * self.nQ)
                    psi = tf.tensor_scatter_nd_update(psi, tf.reshape(index, (1, self.nQ)), [1.0])
                    self.psi = psi

                    for n in range(self.nQ):  # Loop over M
                        self.RotZ(n, line[n])
                        self.RotY(n, line[n])
                        self.RotZ(n, line[n])
                        
                else:
                    
                    #==========================================================
                    # AMPLITUDE EMBEDDING  
                    #==========================================================
                    psi = tf.cast(line, tf.complex64)
                    psi = psi / (tf.norm(psi, keepdims=True) + 1e-12)
                   
                    pad_len = 2**self.nQ - tf.size(psi)
                    pad = tf.zeros([pad_len], dtype=tf.complex64)
                    psi = tf.concat([psi, pad], axis=0)

                    psi = tf.reshape(psi, (2,) * self.nQ)
                    self.psi = psi
                
                #==========================================================
                # CIRCUIT
                self.StrongEntagled()
                
                #==========================================================
                # OUTPUT
                if self.OutType == 1:                    
                    out = self.PauliZ_ExpVal() 
                    
                else:
                    out = self.Probs_TF() 
   
                return out
                
            #==================================================================
                
            return tf.vectorized_map(lambda single_line: process_single_line(single_line), batch)
        
        #======================================================================
        
        return tf.vectorized_map(lambda single_batch: process_single_batch(single_batch), x)
    
    ###########################################################################
    ########################## QUANTUM LAYER CALL #############################
    ###########################################################################
    @tf.function
    def call(self, x):
        x = self.Map(x)
        return x
