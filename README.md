# Predict-real-Disaster-tweets-using-Glove-BiLSTM-self-attention

## Overview

In this notebook I've developed an BiLSTM model to learn the temporal dependencies in the input data and combined it with attention mechanism to focus on the most relevant parts of the input data to predict real disaster tweets from fake ones. I've also used GloVe embeddings in embedding layer in BiLSTM model.

## prerequisites
- Keras
- Glove Embeddings
- [Transformers.BertModel](https://huggingface.co/docs/transformers/model_doc/bert)

### Parameters
- MAX_LEN= 30 (maximum length to pad or truncate the sequences)
- embed_vector_len= 50 (embedding dimetionality)
- batch_train_size= 128
- batch_eval_size= 64
- embed_dim1=64 (dimensionality of the LSTM Layer1 outputs (Hidden & Cell states)- as we have a BiLSTM so the number of units would be 128)
- embed_dim2=32 (Layer2)
- dropout=0.2
- hidden_dim=32 (number of units (neurons) in the Dense layer1)
- num_class=1 
- vocab_size= len(Counter(corpus)) +2
- epochs=100
- optimzer=Adam(learning_rate=1e-3)
- loss='binary_crossentropy'

### Performance
- After 15 epohs:
  - valid-loss: 0.41
  - valid-accuracy: 81 %
