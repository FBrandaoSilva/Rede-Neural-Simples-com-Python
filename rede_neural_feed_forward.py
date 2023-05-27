"""Treinando a rede neural para prever o valor de saída"""

import numpy as np

treinamento_ent = np.array([[0, 0, 1],
                                [1, 1, 1], 
                                [1, 0, 1], 
                                [0, 1, 1]])

treinamento_res = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)

pesos_sinap = 2 * np.random.random((3,1)) - 1

def sigmoid(x):
  return 1/(1 + np.exp(-x))

def sigmoid_derivate(x):
  return x * (1-x)

for i in range(10000):
  input_layer = treinamento_ent
  output = sigmoid(np.dot(input_layer, pesos_sinap))

  erro = treinamento_res - output
  ajuste = erro * sigmoid_derivate(output)

  pesos_sinap += np.dot(input_layer.T, ajuste)

pesos_sinap

print(output)