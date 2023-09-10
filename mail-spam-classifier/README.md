# Spam Detection Project

**IMPORTANT NOTE:** *The goal of this project is to expand and demonstrate skills in machine learning. In this notebook, I will not only explore the possibilities how to create, train and fine-tune NLP models, but also how to use AI tools - primarily LLM - ChatGPT 3.5. Some of the code below was written by AI and then checked and modified by me.*

In this project I will build a machine learning model that classifies emails as spam or ham (not spam). 

To solve this problem I will explore two different methods:

1. F-IDF Vectorization + Multinomial Naive Bayes: In this approach, I will use the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique to convert the email text into numerical features. I will then train a Multinomial Naive Bayes classifier for the classification task.

2. Deep Learning with TensorFlow and Keras: In this approach, I will build a neural network model using TensorFlow and Keras. I will use an Embedding layer to convert text data into numerical vectors, followed by an LSTM layer for sequence modeling and a Dense layer for binary classification.

*I also experimeted and tried to create Support Vector Machine (SVM) classifier with a linear kernel with one-hot encoding the data. This could be seen in section "Method 1.5"*
