#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 08:40:18 2020

@author: Amit


Step 1- Importing requirred libraries  and Dataset

"""
import numpy as np
import tensorflow as tf
import re

import time #This library will measure the training time

"""We can also use pandas for importing the dataset but Here I am using Open Function

This open Function accepts arguements like


File - Either give name of your file (If in the same working directory) or give path of the file

Encoding - If there is encoding issue you can use this arguement, I check my file and it has encoding issue hence will use this parameter

"utf-8"

Errors - Is to ignore any erros popping up

.Read() - This will read the file

.Split (\n) - It is to split the observations by the lines 



"""




lines = open("movie_lines.txt", encoding = "utf-8", errors = 'ignore').read().split('\n')


conversations = open("movie_conversations.txt", encoding = "utf-8", errors = 'ignore').read().split('\n')


"""Now We need to create a dictionary which will map each lines with its id

Now Why do we have to do this? - Look always do this before you jump to any project--- Always you remember where you come from and where you want to go

Yeaah That is great life quote , Anyway! In Coding, Remember this 

Now in this context, We understand where we came from and Now we know where we want to go, Means We need to have a dataset which contains the Input and the Output

Because our Neural network will accept input and predict or show results on Target variable

Now Just imagin in case of ChatBot- It should be something to Reply, Hence our target will be "Reply" replied by the chatBot and Input will be our questions
to this ChatBot. Now to do this ---The Easiest way to do this is to Create a Dictionary

Hence Let's create the dictionary which maps each lines to its ID, Key identifies will be Id and The value will be the line itself


"""

""" Step -2  Now Let's start the Data Preprocessing  """


id2line = {}

for line in lines:
    _line= line.split(" +++$+++ ")
    if len(_line) == 5:
        
        id2line[_line[0]] = _line[4]  
        
        """ Key identifier is our index zero, And Value indentifier is our index 4 hence 0 and 4 """
        
        
        
        """  N
   ow We will create all the list of all the conversations , Now wonder, Why the helll we need to do that?
   
   
   Well We are doing this because we want to keep only relevent data which is important to our training
   , hence we will create  the list of all these lists
   """
   
conversations_ids = []

for conversation in conversations[:-1] :
    
    
    
    """Since we want to loop through our dataset (COnversations) We will name it conversations, 
    
    As we know that Last row of conversations dataset is empty row hence will exclude it 
    
    """
    _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(","))
    
    
    
    

"""Now We will get questions and asnwers  separately

And they will be input and output for our neural network. Questions will be input and Answers will be our output

Now What do we need for this? We need two different lists for Questions and answers separately And Both the lists much be of same size"""


questions= [] 

answers = []

for conversation in conversations_ids :
    for  i in range(len(conversation) -1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])
        
        
"""Now We will do cleaning to the text, To clean the text, I will define a function with a arguement text, And Then assign the tast

to this function like lowering the text

text = re.sub

First arguement in this function is taking word i'm and replacing it with i am, next is in which text we want to replace that hence text

In next line, I am replacing He's by he is and she's by she is etc will do , Let me explain another important replacement here

Pastrofy 'll make it replaced by space will etc. 

Replacing symbolls like @# etc. """

def cleantext(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text) 
    return text


""" Now that We have created this Cleantext function, We will apply it effectly to all the answers and questions

I am going to create a variable called clean_questions which will store cleaned questions, And Run a for loop in questions

Which we already have, It take each of questions inside variable questions  and clean them one by one, 
In Second line, Will apply the clean function which i recently created  And then I  will append this to our new variable 
clean_questions.  And  I will do the same for the answers too"""



clean_questions =[]
for question in questions:
    clean_questions.append(cleantext(question))
    
    
    
clean_answers = []

for answer in answers:
    clean_answers.append(cleantext(answer))
    
    
    
"""Now I will check and remove non frequent words, I will remove only those words which have appeared less than 5% only in the corpus

And To do this, I will create a dictionary which will map each words to its number of occurence, Word2count is a dictionary, Ihave two
separate lists now, Clean questions and clean answers hence will get the words from the both now, Will write a for loop

which will iterate through every words in clean_questions and get their occurence, I will put if condition here to know if

we see the word for the first time or seen it already. If this word is not in word2count. This word gonna get its first ocurrence means count it once

else will increment its number of count or occurences. And I will do the same for answer"""






word2count =  {}

for question in clean_questions:
    for word in question.split():
        if  word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
            
            


for answer in clean_answers:
    for word in answer.split():
        if  word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
            
"""Now I will choose the threshold and count the number of words appear in the CORPUS. I will create two different dictionaries

First dictionary gonna map each words of the questions to unique integers.  and Other dictionary gonna map all the words  

of the answers to  unique intergers, I will write if condition to check if number of occurence of the words is higher than the 

GIVEN threshold. If this is the case, Will include this word in this dictionary.

Other hand if the number of occurences of the words is below the threshold, Will not include it into dictionary


Choosing the threshold as a Number, iN NLP Projects, It is recommended to have at least 5% of the word that appears least in the corpus"""

threshold = 20

questionsword2int = {}

word_number = 0

for word, count,  in word2count.items():
    if count >= threshold:
        questionsword2int[word] = word_number
        word_number += 1
        
        
answersword2int = {}

word_number = 0

for word, count,  in word2count.items():
    if count >= threshold:
        answersword2int[word] = word_number
        word_number += 1
        


""" Now I will add EOS and SOS, wiLL ADD last tokens to these dictionaries """ 

tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]

""" Now let me explain these arguements- If you have gone through the intiution of Seq2Seq model, We have seen that 

There is End of sentence and Start of the sentence

PAD - It is your GPU OR CPU PROCESS THE TRAINING DATA.  in batches and all the sequences in your Batch should have same length
SOS - Start of the sentence
EOS- End of the sentence 
OUT - It is the token by which all the words in our dictionaries will be replaced"""
        
    
for  token in tokens:
    questionsword2int[token] = len(questionsword2int) + 1
    
    
        
for  token in tokens:
    answersword2int[token] = len(answersword2int) + 1
    
    
""" Now I need to create inverse dictionary  of the answersword2int dictionary, Why do we need  do that is because We need this 

While building Seq2Seq model, It is very important to know in Python to create inverse dictionary , to get inverse mapping of the dictionary


And put that  in the dictionary , It is very important because we will be doing often in Deep NLP or DL Projects , Here I am mapping

Answer's integer to word ,, W_i corresponds to word integers means values but other w corresponds to keys """



answersints2word = {w_i: w for w, w_i in answersword2int.items()}



"""Now I need to add EOS to the end of every answers because If you look at the architecture of Seq2Seq we have decoding part which 

we need to end the docoding with EOS tokens, i Again say EOS is needed at the end of decoding layers """


for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'
    
    
    
""" I will translate all the questions and all the answers separately  into integers, Why? Because We have the list of clean questions


Clean answers But i want to convert it into unique integer"""



# Translating all the questions and the answers into integers
# and Replacing all the words that were filtered out by <OUT> 
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionsword2int:
            ints.append(questionsword2int['<OUT>'])
        else:
            ints.append(questionsword2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answersword2int:
            ints.append(answersword2int['<OUT>'])
        else:
            ints.append(answersword2int[word])
    answers_into_int.append(ints)
    
    

# Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            
            
""" Building Architecture of Seq2Seq model"""

""" I am going to create a function which will create placeHolder for the input and target 


Why are we doing so? - Well! In Tensorflow, All the variables are used as Tensors, Tensors are adavanced array and allows the

fasters computation And In tensorflow all the variables should be defined as tensorflow PlaceHolder


We will create statement for inputs and outputs, Input variable where tensorflow placeholder applies 

Placeholder accepts these arguments :
    input_Datatype=(For this project we have converted our input as unique integers)
    
    Dimensions of the matrix of the input data = To get this we can have  a look at our sorted_clean_questions and see that Our 
    Matrix is a 2 dimensional array
    
    Input = This Arguement we have to give is just a Name to the input variable.
    
We will create two more tensorflow placeholder one which will hold learning rate, And Key Prop- It will be used to control the dropout rate

Drop out rate is 20% in general, Hence We deactivate the 20% neurons to avoid overfitting

Lr accepts float32 as its value and name parameter-
    

"""

def model_input():
    inputs = tf.placeholder(tf.int32, [None, None], name = "input")
    targets = tf.placeholder(tf.int32, [None, None], name = "target")
    lr = tf.placeholder(tf.float32,  name = "learning_rate")
    keep_prob = tf.placeholder(tf.float32,  name = "keep_prob")
    return inputs, targets, lr, keep_prob



""" Before we start creating the encoding layer and decoding layer because decoder will only accept certain format of the target

So What is this format for target-

1. It must be in batches (RNN or LSTM won't accept a single target')

Here What do we need to care for?- We need to make sure every sentence in the answers must start with SOS tokens, We can see

our dataset, We have seen that Our answers does not start with SOS tokens hence Need to add SOS tokens in them.

I will add it in the beginning of each answers and each tokens. 

hOW to Add this SOS token at the beggining of each answers

I will take all these answers inside batches and remove the last columns of these answers and take rest of this(Beggining upto the last
                                                                                                                Last column except the
                                                                                                                last column)

And We will make the concatination to add SOS token at the beginning of the target in the batches


Now I will write a function with arguements like Targets, Word2Int and batch_size. 

Left size Will contain the batch size or the line containing only ids of the SOS tokens
I will use tensorflow's fill function because I want to fill the matrix ID with SOS tokens'

It accepts dimensions of the matrix and Value  is basically what do we want to fill in


another statement for thsi function will be right side which is all the answers in the batch except the last one

Will use a tensorflow function in this strided_slice- Let's understand what this function is gonna do

It extracts a subset of tensors. Will extract everything except the last column.

It accepts these arguments

Input = It is your target variable
begin = It will accept the values where from you want to start the extraction
end = It will accept the value you want to end.

slide = Means by how many cells we want to slides when doing the extractions,

Since we want to get all of them except last one hence will give [1,1]

Now Will use pre-process statement  which accepts function "Concat" 

concat accepts values and axis 

If You want to make concatination horizantally Concatination will be 1 
But if you want to make concatination vertically, concatination will be zero


Here We want to make hormizontal concatination, Hence axis= 1


"""




def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets
    




""" Now We have completed our Preprocessing task And ready to create Encoding layer and decoding layers of our Seq2Seq

model architecture 




Creating Encoder  RNN layer """


""" It will not be simple RNN, It will be stakced LSTM with drop out layer 

Will make function which will define somewhere in LSTM


RNN inputs = It will accept the prepared input of the model, Will feed the input which we prepared

RNN_size =  Is the number of input tensor that the encoder RNN LAYER we are making now, 

Number of layers = Num_layers 

Keep_prob = It is dropout. Used to improve the accuracy and prevent overfitting

Sequence_length = It will be a list of the length inside each batch of the questions


***After these arguements Now I will create LSTM****

lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

Then will apply drop out to our LSTM

LSTM_dropout = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob) ***NOTE- This lstm in DropoutWarapper parrenthesis is nothing but the lstm which

we created recently , Keep_prob to control the droput  hence next input is input_keep_prob = keep_prob


Now I have LSTM with droput applied to it. I will create Encoder itself
This encoder_Cell will be multi RNN cell which will get from Tensorflow.

encoder_cell = It will take tensorflow contrib module with some MultiRNN Cell

This Multi RNN Cell will accept some arguements like

It will accept LSTM drop out layer in Square brackets like this-
And to Make it several layer, I will multiply it with by another arguement Number of layers(Num_layers)


Now I have encoder cell, Will get encoder state , Will get it bidirectional RNN functional module

encoder_state  is gonna be returned by bidirectional dynamic rnn function by tensorflow


Encoder state is second state element returned by this BiDirectional Functional,

I will adding _, to specify, Because I only want second element return by this future bidirectional function"""


def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state
 





""" Previously I set up Encoder RNN layer and Now I will set up Decoder RNN layer  

I will do it in three steps-

1. I will decode the training set 
2. Decode the validation set 
3. Finally, Will be creating Decoder RNN layer 


Will be creating a function decode_training_Set for training set- Which accepts arguments like

1. encoder_state - It means decoder is getting encoder state as input
2. Decoder cell -  It is cell in RNN OF THE DECODER
3. Decoder_embedded_input - It is input on which will apply embedding

Embedding is a process of mapping Words to a vector of real numbers

4. Sequence_length -
5. Decoding_scope - 

6. Output_function - Function to be used to return the decoder output at the end

7. keep_prop - Will apply dropout


Now We will write for attention state If you remember, There is Attention machanism used in Seq2Seq model . 

Let me repeat about What is Attention machanism - It is sum of hidden states at the end of every Encoder, And Weighted sum of it called 

Attention machanism, Will initialize them as Three dimensional matrix containing zero. 

I will prepare the attention madule. Can use a function which is there in TensorFlow already to preapre the Machanism module



attention_keys , attention_values, attention_score, attention_construct_function - And Will get all these  from tensorflow's contrib module.Seq2seq Module
.prepare_attention module. 

attention_Keys to be compared to target variables
Attention_Values - Used to construct the context vector
Attention_Score- To compute the similarities between keys and target states
Attention_Construct_Fuction = Is used to get the attention states

Next step - Is to get the training decoder function- THat will get the training decoder states-

Training_decoder_function = Will get from other function in tensorflow which is attention_decoder_fn_train will decode the training




"""

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)



""" 

This function will be same as this above. Only difference is that I created for training set and This I will create for Test/Validation 

set"""


def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions




    
    

    """ After creating the components used within Decoder, Now I will create Entire Decoder of Seq2Seq model, It is very similar 
    
    encoder rnn, Here Will create decoder_rnn function- Will have some arguements, I will not explain all of these arguments
    because It has been implemented in encoder
    
    1. Decoder_embedded_input
    2. decoder_embedding_matrix
    3. Encoder_State - It is the out of the encoder and because input to the decoder
    4. number_words = Will take totak number of words in the corpus. 
    5. Sequence_length
    6. rnn_Size
    7 numb_layers = It is number of layers we want to have inside of decoder
    8. word2int dictionary
    9. keep_prob = Drop out will applied
    10. Batch_size = """ 
    

def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
    

""" Building Seq2Seq model, I have built every components which were requirred to build Seq2Seq model 


Function Seq2Seq will accept arguments

input = Questions from corpus
Target = Answers from corpus
keep_prob
batch_Size
Sequence_length
numb_words (for answers and questions)
encoder_embedding_Size - Dimensions of the embedding matrix
decoder_Embedding_Size
rnn_size 
number_layers in decoder cell - Num_layer

questionsword2int - It is dictionary to preprocess the target

Now I will assemble that will return the encoder state and decoder state




"""



def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions





""" Training the Seq2Seq model 

Setting the Hyper parameters for model

epochs = 100, It is one whole iteration of the training
batch_size = 65, Batche will be fed to Model
rnn_Size = 512
number of layers = 3 How many layers do we want in encoder RNN, and Decoder RNN?
encoding_embedding_size = It is number of column in embedding matrix. 512 columns in embedding matrix
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay =  It is by which percentage the learning rate is reduced. 0.9

minimum of the learning_rate = It is to be written because I am applying decay here, 0.0001

keep_probability =  0.5"""

# Setting the Hyperparameters
epochs = 100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5

""" Defining a session ,    On which all the tensorflow training will be run
Will create an object 
Will reset tthe graph functuon which resets the tensorflow default graph"""

tf.reset_default_graph()
session = tf.InteractiveSession


""" Will laod the input to our seq2seq model, You have remember I have already made the model_input function  """
 
# Loading the model inputs
inputs, targets, lr, keep_prob = model_input()
 
"""
Setting the sequence length"""

sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')


"""
Getting the shape of the inputs tensor 

"""

input_shape = tf.shape(inputs)



 
"""Getting the training and test predictions"""


training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answersword2int),
                                                       len(questionsword2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionsword2int)


"""Setting up the Loss Error, the Optimizer and Gradient Clipping"""

with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
 
 

"""Padding the sequences with the <PAD> token"""


def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]



"""Splitting the data into batches of questions and answers"""



def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch




"""splitting the questions and answers into training and validation sets"""


training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

    

"""Training"""

batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt" # For Windows users, replace this line of code by: checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")
