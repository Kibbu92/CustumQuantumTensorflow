import pennylane as qml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import time 

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist

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
class QNN(nn.Module):
    
    def __init__(self, nQ, nC):
        
        super(QNN, self).__init__()        
             
        self.conv1 = nn.Conv2d(1, 4*nQ, kernel_size=(2, 2), padding='same')
        self.conv2 = nn.Conv2d(4*nQ, 2*nQ, kernel_size=(2, 2), padding='same')
        self.conv3 = nn.Conv2d(2*nQ, nQ, kernel_size=(2, 2), padding='same')        
        self.pool  = nn.MaxPool2d((3, 3), stride=(1, 1))

        self.flatten = nn.Flatten()           
        self.quantum_layer = ql_qml  
        self.fc = nn.Linear(22*22*nQ, nC)  # Fully connected layer to output `n_class` classes
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        a,b,c,d = x.shape
        x = torch.reshape(x, (a,b,c*d)).permute(0, 2, 1)        
        x = self.quantum_layer(x)  
        x = torch.flatten(x, start_dim=1)        
        x = self.fc(x)
        
        return F.softmax(x, dim=1)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::    
class CustomCategoricalCrossentropy(nn.Module):
    def __init__(self, eps=1e-10):
        super(CustomCategoricalCrossentropy, self).__init__()
        self.eps = eps  

    def forward(self, y_pred,y_true):
        y_pred = torch.clamp(y_pred, min=self.eps, max=1.0 - self.eps)        
        loss = -torch.sum(F.one_hot(y_true) * torch.log(y_pred), dim=-1)        
        return torch.mean(loss)    

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

    X_train = np.expand_dims(X_train, axis=-1)
    X_test  = np.expand_dims(X_test , axis=-1)
    X_valid = np.expand_dims(X_valid, axis=-1)
    
    ###########################################################################
    nQ = 4    
    nL = 1 
    nC = 10    
    
    dev    = qml.device("default.qubit", wires=nQ)      
    Qcirc  = qml.QNode(quantum_circuit,dev,interface='torch') 
    ql_qml = qml.qnn.TorchLayer(Qcirc, weight_shapes={"weights": (nL, nQ, 3)}) 
    
    ###########################################################################
    LR = 1e-3
    Epoch = 10
    Batch = 100 
    
    ns = 10
   
    X_trn, Y_trn = X_train[::ns], Y_train[::ns]
    X_val, Y_val = X_valid[::ns], Y_valid[::ns]  
    
    ###########################################################################   
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = QNN(nQ=nQ, nC=nC)  # You can change the number of qubits (nQ) and number of classes
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-7)  
    Loss_Funciton = CustomCategoricalCrossentropy()
    
    ###########################################################################
    # Dataset Initialization for Pytorch Implementaiton
    X_train = torch.tensor(X_trn, dtype=torch.float32).permute(0, 3, 1, 2).to(device) 
    Y_train = torch.tensor(Y_trn, dtype=torch.long).to(device)  
    X_valid = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2).to(device) 
    Y_valid = torch.tensor(Y_val, dtype=torch.long).to(device) 
         
    train_dataset = TensorDataset(X_train, Y_train)
    valid_dataset = TensorDataset(X_valid, Y_valid)
    
    train_loader = DataLoader(train_dataset, batch_size=Batch, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Batch, shuffle=False)
    
    ###########################################################################  
       
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    for i in range(Epoch):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        T0 = time.perf_counter() 

        for batch_idx, (data, target) in enumerate(train_loader):
            # data, target = data.to(device), target.to(device)  # Send data and target to the GPU (if available)
            
            optimizer.zero_grad()             
            output = model(data)            
            loss = Loss_Funciton(output, target)            
            loss.backward()            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output, 1)
            correct_predictions += (predicted == target).sum().item()
            total_samples += target.size(0)      
            
        TF = time.perf_counter()        
        DT = (TF-T0)/len(X_trn)
        
        avg_loss = running_loss / len(train_loader)
        train_accuracy  = correct_predictions / total_samples * 100
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        
        model.eval()
        valid_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        with torch.no_grad():
            for data, target in valid_loader:
                    data, target = data.to(device), target.to(device)
                    
                    output = model(data)                    
                    loss = Loss_Funciton(output, target)
                    valid_loss += loss.item()
                    
                    _, predicted = torch.max(output, 1)
                    correct_predictions += (predicted == target).sum().item()
                    total_samples += target.size(0)
                
            avg_valid_loss = valid_loss / len(valid_loader)
            valid_accuracy = correct_predictions / total_samples * 100
            
            valid_losses.append(avg_valid_loss)
            valid_accuracies.append(valid_accuracy)
            
            print(f"({DT:.2e}) Epoch {i} - Loss: {avg_loss:.4f}, Accuracy: {train_accuracy :.2f} - Validation Loss: {avg_valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")
        
    # torch.save(model.state_dict(), 'quantum_nn_model.pth')

    results = {
        "train_loss": train_losses,
        "valid_loss": valid_losses,
        "train_accuracy": train_accuracies,
        "valid_accuracy": valid_accuracies
    }
    
    results_df = pd.DataFrame(results)
    
    results_df.plot(y=["train_loss", "valid_loss"], title="Loss vs. Epochs", xlabel="Epochs", ylabel="Loss")
    plt.grid(True)
    plt.show()
    
    # Plot delle accuratezze
    results_df.plot(y=["train_accuracy", "valid_accuracy"], title="Accuracy vs. Epochs", xlabel="Epochs", ylabel="Accuracy (%)")
    plt.grid(True)
    plt.show()
       
     
        
            
        
        
            
          
