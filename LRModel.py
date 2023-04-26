import torch
from torch import nn
from sklearn.model_selection import train_test_split #Selección aleatoria de muestras
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data importation
dataframe = pd.read_csv("datosModelo.csv")
X = [] #Trajectories vector
y = [] #Labels vector
ws = [] #Works vector

for i in range(1000):
    X.append(np.array(dataframe["Tr_{}".format(i)]))
    y.append(int(dataframe["Label_{}".format(i)][0]))
    ws.append(float(dataframe["W_{}".format(i)][0]))
    
X,y,ws = np.array(X),np.array(y),np.array(ws)

#Data arrangement
XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size=0.2,random_state=42)

#Re-scaling data 
sc = StandardScaler()
XTrain = sc.fit_transform(XTrain)
XTest = sc.transform(XTest)

#Converting data into torch tensors
XTrain = torch.from_numpy(XTrain.astype(np.float32))
XTest = torch.from_numpy(XTest.astype(np.float32))
yTrain = torch.from_numpy(yTrain.astype(np.float32))
yTest = torch.from_numpy(yTest.astype(np.float32))

#Re-scaling y data
yTest = yTest.view(yTest.shape[0],1)
yTrain = yTrain.view(yTrain.shape[0],1)

#Printing device information
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#Model implementation
class LogisticReg(nn.Module):
    def __init__(self, inputDim, outputDim):
        super().__init__()
        self.linear = nn.Linear(inputDim, outputDim)
        
    def forward(self,x):
        prediction = torch.sigmoid(self.linear(x))
        return prediction
        
model0 = LogisticReg(inputDim = 1001, outputDim = 1).to(device)

#Loss and optimization function
learningRate = 0.0001
lossFunction = nn.HingeEmbeddingLoss()
optFunction = torch.optim.Adam(params = model0.parameters(),lr = learningRate)

#Precission function
def precissionFunc(yReal,yPred):
    right = torch.eq(yReal,yPred).sum().item()
    prec = (right/len(yPred))*100
    return prec

#Learning loop
epochs = 10000
epochList = []
trainLoss = []
testLoss = []

for epoch in range(epochs):
    #Training mode
    model0.train()
    
    #Prediction making
    yPred = model0(XTrain)
    
    #Prediction's loss
    loss = lossFunction(yPred,yTrain)
    
    #Zero grad
    optFunction.zero_grad()
    
    #Backpropagation
    loss.backward()
    
    #Takes a step
    optFunction.step()
    
    #Model evaluation
    model0.eval() #Converts to evaluation mode
    with torch.inference_mode(): #Stops monitoring gradients 
        
        testPred = model0(XTest)
        
        #Takes the loss of testing
        lossT = lossFunction(testPred,yTest)
        testLoss.append(lossT)
        
    #Se guardan los datos para graficar
    epochList.append(epoch)
    trainLoss.append(loss)
    
    if epoch%100 == 0:
        print("Epoch : {}, Training Loss : {}, Test Loss : {}".format(epoch,loss,lossT))

#Plotting training loss and test loss
plt.figure(figsize=(8,8))
plt.plot(epochList,trainLoss,label="Training loss")
plt.plot(epochList,testLoss,label="Testing loss")
plt.xlabel("Epochs",fontsize=14)
plt.ylabel("Loss",fontsize=14)
plt.title("Test and training loss for the model. LR = {}".format(learningRate),fontsize=15)
plt.legend(fontsize=13)
plt.savefig("LossEpochs_ADAM_HE.png")
plt.show()

#Saving the model
from pathlib import Path

modelRoute = Path("Modelos") 
modelRoute.mkdir(parents = True, exist_ok=True) 

modelName = "modeloProyecto1000Tr.pth" 
savedModelRoute = modelRoute/modelName 

torch.save(obj=model0.state_dict(),f = savedModelRoute) 