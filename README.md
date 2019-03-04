# mnist from scratch
This repository contains code to categorise a set of mnist jpegs in to their 10 categories using a neural network. The code is written in python. Its dependencies are numpy for mathematical functions such as np.exp() and h5py for outputting checkpointing.

This project's aim is to explore the mathematical foundations of back-propagation and gradient descent. The mathematics used in the code can be summarised by four equations which can be found on [Michael Nielsens blog](http://neuralnetworksanddeeplearning.com/chap2.html).

Notation:

The error of neuron j in layer l is given by:

![equations](/images/error_eq.png)

where Z is:

![equations](/images/z_eq.png)

Note: when the activation function is applied to Z_l, it then becomes the activation for layer l



![equations](/images/tikz21.png?raw=true)

Where L is the final layer in the network, and l is the lth layer in the network starting from the input layer being l = 0.


## Setting up
 
create a virtual environment for the project, and install numpy and h5py.

```
$ pip install numpy
```

```
$ pip install h5py
```

## Data
In the spirit of doing things from scratch, the starting point of this project is jpegs, which is one of the most widely used image formats.

Jpeg mnist images can be found at this [link](https://www.kaggle.com/scolianni/mnistasjpg) on kaggle.
It is useful to set up an account on kaggle and to use the kaggle API to download the data set.

## Setting up a kaggle account
Once you have set up a kaggle account, go to "my profile" > "profile" > click on "create a new API token". This will automatically start a download of a json file called kaggle.json
Move this to your home directory under the directory name ".kaggle"

```
$ mv kaggle.json ~/.kaggle/
```
Then install kaggle
```
$ pip install kaggle
```

Now we are able to download the data set from kaggle.
This project relies on the mnist data being categorised in to files named 0 to 9, each containing images of those numbers.

```
$ kaggle datasets download -d scolianni/mnistasjpg
```

## Run the training
```
$ python mnist_fully_connected.py
```
