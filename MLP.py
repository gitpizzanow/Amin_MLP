import numpy as np

"""

binary cross entropy  +  sigmoid

forward


"""





class Layer :
  
  def __init__(self,input_size , output_size):
    self.input_size  = input_size
    self.output_size = output_size
    
    self.weights = np.random.uniform(-1,1 , (output_size,input_size))
    self.baises  = np.random.uniform(-1,1 ,(output_size,))
    self.outputs = np.zeros((output_size,))
    
    self.deltas = np.zeros((output_size,))
    
    self.dw  = np.zeros_like(self.weights)
    self.dB  = np.zeros_like(self.baises)
    
    self.next = None
    #self.prev = None
    



def sigmoid(z) :
  return 1.0 / (1.0 + np.exp(-z))

def step(a) :
  return 1 if a>=0.5 else 0

def derivative(a) :
  return a * (1-a)

def cross(y , yhat):
  yhat = max(yhat , 1e-7) # 0.0000001
  yhat = min(yhat , 1.0 - 1e-7)
  return -(y * np.log(yhat) + (1-y) * np.log(1-yhat))



#forward  z = wx + b

def forward(
input : np.ndarray,
layer ) :
  #Z = w @ input
  z = layer.weights @ input
  
  
  z +=  layer.baises.reshape(-1,1)
  layer.outputs = sigmoid(z)
  
  layer.outputs = layer.outputs.flatten()
  


def compute_deltas():pass
  





if __name__ == "__main__":

  hidden1 = Layer(3,4)
  hidden2 = Layer(4,4)
  output_layer = Layer(4,1)
  
  hidden1.next = hidden2
  hidden2.next = output_layer
  
  layers : list[Layer] = [hidden1,hidden2,output_layer]
  N_layers = len(layers)
  
  X = np.array([ #XOR
    [1,0,1],#0
    [1,1,1],#1
    [1,0,0],#1
    [1,1,0],#0
    [0,0,1],#1
    [0,1,1],#0
    [0,0,0],#0
    [0,1,0],#1
    ] , dtype = np.float32) # (8,3)
  
  y = np.array([0,1,1,0,1,0,0,1],dtype = np.float32).reshape(-1,1)
  
  
  for i in range(X.shape[0]): # 8
    input = X[i].reshape(-1,1) #1 * 3
    for layer in layers: ## 3 
      forward(input ,layer)
      input = layer.outputs.reshape(-1,1)
    print(step(layer.outputs[0]))

