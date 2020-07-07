# BERT and A List of NLP frameworks (NLP Infographic)

# NOTE :- Make sure You are aware about the functionality of Transformer before reading it.

> # Agenda of this Notebook
> BERT architecture

> Text preprocessing

> Pre-Trained taks

> Implementation of the BERT

> A List of NLP Framework (NLP Infographic)



BERT stands for Bidirectional Encoder Representations from Transformer. It has two variants available.
- **BERT Base** : 12 Layers (Transformer blocks), 12 Attention heads and 110 million parameters
- **BERT large** : 24 Layers (Transformer blocks), 16 Attention heads and, 340 million parameters

BERT is the first architecture that jointly fuses the left and right context in all layers. Previous state of art architecture takes only Right to left or Left to right.

** How does BERT work **

Let's deep dive into working of the BERT

BERT expects the tokens to be converted into lower case,  tokenised using wordPiece, Index based on Vocabulury file provided by BERT. Sentence pair  is split with token [SEP]. The WordPiece embeddings are added with segment embeddings, Which marks the first sentence as A and Second Sentence as B, And the positional embeddings for each word. There will be another special classification embeddings tokens [CLS] is added at the beginning of every  input.

Let me divide the few terms here into part-

- **Position embeddings** : BERT learns and uses positional embeddings to express the position of words in a sentence. These are added to overcome the  Limitation of Transformer.
- **Segment Embeddings** : BERT can also take sentence pairs as input  for tas (Question and Answer).
- **Token Embeddings** : These are embeddings learned for the specific token from the WordPiece token vocab

### BERT is pretrained on two NLP Tasks

- Masked Language Modeling
- Next sentence Prediction

** A). **  Masked Language Modeling - (Bi-Directionality) 

BERT is designed as deeply bidirectional model. The network captures infromation from both the right and left  context of a token from the first layer itself. This is where Masked Langugage model comes into the picture.

**What is Mask Language model?**  Masked language modeling is a fill in the blank task. Where a Model uses the context words surrounding a [[MASK]] Token to try to predict the [[MASK]] word should be. MASK is nothing but missing word. Mask model randomly masks 15% tokens from an input sequence and train on the model to predict  the original vocab id of the Masked word based on it context.

BERT masked randomly 15% of the words to prevent model focusing too much on  particular position or tokens are masked. Mask token would never appear during the training hence Masked words were not always replaced by the masked tokens.

80% of the time, Words were replaced with the masked token
10% of the time, word replaced with random words
10% of the time, Word were left unchanged.


** B). ** **Next Sentence Prediction** - BERT is also trained on NLP tasks which requirred an understanding of the relationship between the sentences. Next Sentence Prediction REPLACES THE SECOND INPUT SENTENCE WITH A RANDOM SENTENCE  FOR 50% OF THE TRAINING steps.  and trains on other sentence pairs to learn sentence relationship. 



This is How BERT outperforms the other models 

It is Bidirectional Model which captures the information right and left context.
It uses Mask Langugage model
It uses Next Sentence Prediction



## NLP Infographic 

- Transformer Architecture developed by Google AI
- Universal Language Model Fine tuning (ULMFIT) developed by FAST.AI [Research Paper](https://arxiv.org/pdf/1801.06146.pdf)
- BERT Devloped by Google AI 
- Google's Transformer-XL - It outpeformed BERT in LM and also resolved Transformer's Context fragmentation.
- OpenAI's GPT-2
- XLNet - Auto-Regresive method for LM. Best for both BERT and Transformer-XL developed by CMUAI
- PyTorch Transformer- Pre-Trained SOTA model and Fine Tuned SOTA model.
- Baidu's developed by Baidu Research
- ROBERTA(Optimized pretraining model)  developed by Facebook Research Team. An improvement over BERT
- Spacy-PyTorch transformers developed by Spacy
- FacebookAI's XLM/mBERT by Facebook Research Team
