# SNNvC-
This is SNN(spiking Neural Network) made using C++.

#Train method.

The base method is STDP(spike timing dependent plasticity), which is unsuperviesd learning using time difference between post- and pre-neuron. I used neuron potenrial voltage change to find the correlation of SNN train inculding the difference. Also, I applyed reinforcement learning and self-supervised learning to this model.

# Why do you change the training method?

In my perspective, for improving accuracy, it is the most important thing that the neuron's weight must have different shape, like number 1 has two shape, "l" or "1". The reinforement learning guides neurons to have the correct weight values and self-supervised learning can be the clue about how to increase the Unsupervised SNN's model accuracy.

# How to apply?
First, I used the STDP duing 1 epoch. Then, if some neurons have the same index during the time, they are trained by the combination learning.  In this case, if the neuron answers the neuron label, the neuron can be trained by the reinforcement learning. But if not, the neuron can not be done. In my model, to increase the accuracy, I trained the model when the neuron can not get the answer. In this point, I applied the self-supervised learning because the different number number have the common pixels. So, the neuron can be trained itself using own weight and the results showed the improvement of accuracy 5%.

#Before

![420000 3](https://user-images.githubusercontent.com/86340022/230850252-aeda8eaf-1717-4ad9-a79f-37b6521bb724.jpg)

#After

![300000 3](https://user-images.githubusercontent.com/86340022/230850219-98149c15-0c29-40ed-9e8e-942dc58fd377.jpg)
