import pennylane as qml
import jax.numpy as jnp
import haiku as hk
import jax
import itertools

from typing import  Tuple

class QCNN(hk.Module):
    def __init__(self, 
                 knl: Tuple = (2, 2), 
                 nL: int = 4):
        
        super(QCNN, self).__init__()      
        self.nQ   = knl[0]*knl[1]
        self.nL   = nL
        self.knl  = knl
        
        dev = qml.device("default.qubit", wires=range(self.nQ))
        qnode = qml.QNode(self.circuit, dev)
        circ  = jax.vmap(qnode, (None, 0)) 
        circ  = jax.vmap(circ, (None, 0))
        self.circ = circ 
         
    def circuit(self, angles, data):
        
        for j in range(self.nQ):
            qml.Rot(jnp.pi * data[j, 0], 
                    jnp.pi * data[j, 1], 
                    jnp.pi * data[j, 2], 
                    wires=j)

        for il in range(self.nL):
            # rotations
            for iq in range(self.nQ):
                qml.RX(angles[il, iq, 0], iq)
                qml.RY(angles[il, iq, 1], iq)
                qml.RZ(angles[il, iq, 2], iq)
            
            for ie in range(1, self.nQ, 2): # odd
                qml.CZ(wires=[ie-1, ie])
            
            for ie in range(2, self.nQ, 2): # even
                qml.CZ(wires=[ie-1, ie])       
       
        return [qml.expval(qml.PauliZ(i)) for i in range(self.nQ)]
    
    ###########################################################################    
    def PreProc(self,images):
        
        fun = lambda image, start_point: jax.lax.dynamic_slice(image, start_point, list(self.knl)+[3])     
        slice_im    = jax.vmap(fun, (None, 0))
        slice_im_ij = jax.vmap(slice_im, (0, None))    
        
        dim_i = images.shape[1] - self.knl[0]//2
        dim_j = images.shape[2] - self.knl[1]//2
        
        comb = jnp.array( list( itertools.product(jnp.arange(dim_i), jnp.arange(dim_j)) ) )
        O = jnp.zeros((comb.shape[0], 1), dtype=int)    
        comb = jnp.hstack((comb, O))
        result = slice_im_ij(images, comb)
        
        new_shape = list(result.shape)  
        new_shape[2] *= new_shape[3]  
        del new_shape[3]  
        
        result = result.reshape(new_shape) 
        
        return result
    
    def __call__(self, x):
        
        W_init = hk.initializers.RandomUniform(-jnp.pi, jnp.pi)
        W = hk.get_parameter("angles", shape=[self.nL, self.nQ, 3], dtype=x.dtype, init=W_init)
        
        x = self.PreProc(x)
        x = self.circ(W, x)        
        x = jnp.stack(x)
        x = jnp.moveaxis(x, 0, -1)
     
        return x