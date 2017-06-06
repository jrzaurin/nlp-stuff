# Text-Classification-with-Tensorflow

Part of the code in this repo is based on this great [keras tutorial](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html) by F. Chollet, which I recommend reading. 

The aim here is simply to illustrate 3 different ways of building a Convolutional neural network for text classification using Tensorflow. 

The approach is described in the post I mentioned above, and consists mainly in using word embeddings (the well known GloVe embeddings) with a series of 1D Convolutions. This is the architecture:

`[Emb layer + Emb Look-up] --> 3x[Conv1D + MaxPool1D + BN] --> [FC] --> [FC_output]`

In addition, global max pooling is also used. This means that in the output of the last convolutional layer, every unit will encode all the information coming from one filter. Let’s go through the numbers to make this clearer: 

1- Each document is represented by word-sequences of 1000 words max. Within each sequence, each word is represented by its embeddings, in this case, of 100 dimensions. Therefore, each document will be represented by an array of dim `(1000, 100)`. For a given batch size, the input of the first convolutional later will be a tensor of dim `(batch_size, 1000, 100)`.

2- We then use  convolutional layer with 128 filters, a receptive field of 5 (i.e. filter size = 5) and a stride of 1. This, with `VALID` padding (i.e. no extra padding added to accommodate the filter dimensions) results in a tensor of dimensions `(batch_size, 996, 128)`

3- We then use a max_pool layer with kernel_size and stride of 5, which results in a tensor of dimensions `(batch_size, 199, 128)`

4- Doing the same through the 3 convolutional layers we see that after the last convolution we end up with a tensor of shape `(batch_size, 35, 128)`, where every filter’s information is encoded in 35 numbers. At this stage, we apply the global max pooling, with a kernel size of 35, resulting in a tensor of `(batch_size , 128)`  where each one of the 128 dimensions encodes the information from one filter in that last convolutional layer. 

5- Finally, the 128 outputs are “plugged” into a fully connected layer with 128 neurons and from there to the last output layer with as many neurons as classes (20).

I have used three different utilities within Tensorflow to build the architecture. From lower to higher level:

1-“Pure” Tensorflow, using mainly `tf.nn` 

2- The `layer` module along with the `model_fn` library.

3-The high level API `tflearn`, which I guess should be the "go-to" tool. 

You will need the 20newsgroup dataset and the GloVe vectors to run the scripts. You can get them from here: 
* [20newsgroup](http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html)
* [Glove](https://nlp.stanford.edu/projects/glove/)

if you are going to run the scripts interactively, remember to use `tf.reset_default_graph()` each time you re-define your graph. 

Running a script takes around 150sec using an AWS p2xlarge instance (with a Tesla k80 GPU)

NOTE: The author claims to obtain 95% accuracy on validation. When using the code [here](https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py) we never managed to obtain more than around 73%. This is consistent with the Tensorflow results. Nonetheless, I will emphaize that the aim of the code here is to illustrate the variety of tools within Tensorflow.  
 
 
