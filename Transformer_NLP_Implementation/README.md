# Deep_NLP_Projects
# Project - 1 - Application of Transformer in Deep NLP



## Challenges and Experiments with transformer :

For the first time, I got good accuracy but transformer's result was not satisfying. Whatever input i used to give to the transformers. It kept giving the sentence with same words.

Later I found out that it is because in Scale.product forgot the minus sign that was not supposed to there hence what is did was 

It multiplied the mask by zero instead of minus infinity so Look ahead mask which i built was not applied,  and Decoder allowed itself to have a look at  the future words instead of guessing them




