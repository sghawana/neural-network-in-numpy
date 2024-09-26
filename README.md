# neural-network-in-numpy
I made a Multi Layer Perceptron(Neural Network) in numpy

---
### Backpropogation in Neural Networks


Backpropagation is an algorithm used for computing gradients of the loss function with respect to weights in the neural network. The gradient is then used to update weights via gradient descent.


For a single linear layer, we calculate the gradient of the loss  $L$  with respect to the parameters  $W$ ,  $b$ , and the input  $x$ . Let’s denote the output of the linear layer as  $y = Wx + b$ , and the loss as  $L$ .

**1. Gradient of the Loss with respect to Output  y:**
The gradient of the loss with respect to the output of the current layer is passed down from the next layer in the backward pass:
$$ \frac{\partial L}{\partial y} $$

**2. Gradient with respect to Weights  W :**
Using the chain rule, the gradient of the loss with respect to the weights is computed as:
$$ \frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial W} = \delta \cdot x^T $$
Where  $\delta = \frac{\partial L}{\partial y} $ represents the gradient propagated from the next layer.

**3. The gradient with respect to bias is calculated as:**
$$ \frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} = \delta $$

**4. Finally, the gradient of the loss with respect to the input to this layer is:**
$$ \frac{\partial L}{\partial x} = W^T \cdot \delta $$

**Activation Function’s Gradient**

If the activation function  \sigma  is applied after the linear transformation, we need to propagate gradients through it. For an activation function  \sigma(y) , the gradient is:

$$ \frac{\partial L}{\partial z} = \frac{\partial L}{\partial y} \cdot \sigma’(y) $$

where  $\sigma{\prime}(y)$  is the derivative of the activation function.

**Full Backpropagation Process**

For each layer during backpropagation, we perform the following steps:

1.	Calculate the gradient of the loss with respect to the output (from the next layer).
2.	Compute the gradients with respect to weights, biases, and inputs using the chain rule.
3.	Update the parameters using gradient descent:
$$ W \leftarrow W - \eta \cdot \frac{\partial L}{\partial W} $$
$$ b \leftarrow b - \eta \cdot \frac{\partial L}{\partial b} $$

where  $\eta$  is the learning rate.


---
## Implementation

### 1. Linear Layer


The Linear Layer in a Multi-Layer Perceptron (MLP) computes the following operation:

$$ y \ = \ \sigma (Wx + b)$$

where:

•	 $W$  is the weight matrix,
•	 $x$  is the input,
•	 $b$  is the bias term, and
•	 $\sigma$  is the activation function applied element-wise.


```
layer =  Linear(input_size, output_size, activation, use_bias,
              dropout, use_act, eval, regularisation, beta)
```

1. Input size (int) : Size of input dimension ($X$)

2. Output size (int) : Size of output dimension ($Y$)

3. Activation ['relu', 'sigmoid']: Type of non Linearity **[default : Sigmoid]**

4. Dropout (float) : fraction of nodes turned off using dropout **[default : 0]**

5. Use Activation (bool) : To use activation ($\sigma$) in the layer or not **[default : TRUE]**

6. Use Bias (bool) : To use bias $b$ or not **[default : TRUE]**

7. Eval (bool) : When true layer store gradients for training, false for inference **[default : TRUE]**

8. Regularisation ['l1', 'l2'] : Type of regularisation  **[default : none]**

9. beta (float) : Regularisation constant $\beta$ **[default : 0]**

---

### 2. Multi Layer Perceptron 

```
model =  MLP(input_size, output_size, hid_sizes,
          activation, is_softmax, reg, beta, layer_dropout,
          use_bias, eval)
```

1. Input Size ( int ) : Number of nodes in input layer i.e size of input

2. Output Size ( int ): Number of nodes in output layer i.e size of output

3. Hidden sizes ( list[ int ] ) : Size of each hidden layer between input to output layer (in order) ; **[default : [  ] i.e. no hidden layer]**

4. activation : Type of non Linear activation used in Neural Network **[default : sigmoid activation]**, (same for all layers)

5. Is_softmax (bool) : To use softmax at output of Neural Network **[default : False]**

6. Layer Dropout (float) : Dropout ratio for layers in Neural Network **[default : 0]** (same for all layers)

7. Use Bias (bool) : To use bias $b$ or not **[default : TRUE]** (same for all layers)

7. Eval (bool) : When true model is in traning mode, false for inference **[default : TRUE]**

8. Regularisation ['l1', 'l2'] : Type of regularisation  **[default : none]**

9. beta (float) : Regularisation constant $\beta$ **[default : 0]**

---

### 3. MLP Trainer

#### Training
```
trainer =  MLPTrainer(model, learning_rate,batch_size , lossfn)
trainer.train(x_train, y_train, x_val, y_val, epochs)
```

1. Model ( object ) : Instance of a MLP class

2. Learning Rate ( int ): Step size in stochastic gradient descent

3. Loss Function ['sq-error', 'cross-enropy']: Loss function for regression or classification

4. Batch Size ( int ): Batch size for Training


To get batches separately for features and Labels for training and validation data
```
batches = trainer.data_loader(data)
```
#### Inference

```
y = model.forward(x)
```
---


