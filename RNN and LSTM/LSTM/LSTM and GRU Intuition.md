# Deep Learning- Deep Dive into Deep Learning


# How does LSTM and GRU function?


![lstm_2.svg](file:///Users/kris/Library/Containers/net.jocius.reflection/Data/Documents/Reflection/J1/2020/07/Files/lstm_2.svg)

** Dependencies**

## The Previous cell state- 

The information that was present  in the memory after the previous  time step

## The Previous hidden state - 

This is same as the output  of the previous cell

##The input at the current time step -

 New information being fed in at that moment


**Main core concepts of LSTM are GATES AND CELL STATE**

It has 3 gates and 1 cell state.

**Forget gate**  - How much of the previous comput state for the current input you want to let through.

**Cell state** 

**Input gate**  It defines how much of the newly computed state for the current input you want to  let through. 

**Output gate**  It states How much of the internal state you want to expose to the external network

These gates contains **Sigmoid** activation function. It is similar to the tanh. It squashes values between zero and 1. But tanh squashes values between -1 to 1. 



Let's how does it work!

Previous hiddent state and input get concatenate and This concatanatation called combine
 information is passed through Forget gate decides how much information should be passed in and kept away. It takes information from previous state and information from the current input(Is multiplied by weight matrix) is passed through a Sigmoid function. values come out between zero and one. A Closer to zero means forget and a closer to one means to keep, And it updates  the cell state accordingly. And then This Sigmoid values vector is multiplied with Cell state.


We have the input Gate. It is reponsible for adding the information to cell state (Look closely at the + sign) , we pass the previous hidden state and the current input to the sigmoid function that decides which values will be updated by transforming values between zero and one. It also passes the hidden state and current input  to the **Tanh** function to switch values between  -1 and 1  then multiply the **Tanh** output with **Sigmoid output**. Now at this stage we will have enough information to the cell state via addition operation.


Output Gate will accept the hidden state and input words from the corpus at its input, It will apply  sigmoid function and output of the sigmoid function will be multiplied  by the output of the Tanh function (This Tanh function accepts  Cell state information as input) and Whatever information we get  will be hidden state to other Cell and there will be some Information from cell state too. 

As a final result , We will accept Cell state information as Final words




- LSTM can cope with Vanishing Gradient descent problem. 

- It can memorise the words for longer time of period than RNN



# GRU 

GRU has two gates reset gate and update gates

- Reset gate 

It is used to decide how much pass information to forget.

-Update gate 

Is similar to input and forget gate of LSTM it decides what information to throw away and what information to keep.

Benifits-

GRU has less tensor operation hence they are faster than LSTM



# End notes 

- RNN Suffers from Short term memory and has Vanishing gradient problem also. Sequential processing made it difficult to learn long term dependencies.

- LSTM uses gate to fix the RNN's problem and Gates are just neural network to regulate the flow of the information, In LSTM words are passed sequentially and generated sequentially. They learn left to right and right to left and combining them. But Transformer architecture addresses this concern and  they accept every words in corpus at once. They are faster. They can learn context of words from both side. 

- GRU has less tensor operation hence they are faster than LSTM

LSTM and GRU are used for speech recognition, speech thensis


Reference links 

[Colah Blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs)

[WildML Blog](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano)





## BERT - BiDirectional Encoder Representation from Transformer


It is a pre-trained NLP model (Transformer). Which uses Self Attention machanism. 
Selft Attention is a machanism which allows words to learn relationship with all other words. 

**Types of Attention** -

Encoder self attentions and Decoder self attention. Let's you are building a translator from french to English. In this case. Encoder will deal with french and Decoder will deal with English sentence


