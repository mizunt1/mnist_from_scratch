# mnist from scratch
This repository contains code which categorise a set of mnist jpegs in to their 10 categories using a neural network written in python.

## Data
In the spirit of doing things from scratch, the starting point of this project is jpegs, which is one of the most widely used image formats.

Jpeg mnist images can be found at this [link](https://www.kaggle.com/scolianni/mnistasjpg) on kaggle.
It is useful to set up an account on kaggle and to use the kaggle API to download the data set.

## Setting up a kaggle account
Once you have set up a kaggle account, go to "my profile" > profile > click on "create a new API token". This will automatically start a download of a json file called kaggle.json
Move this to your home directory under the directory name ".kaggle"

```
$ mv kaggle.json ~/.kaggle/
```
Then install kaggle
```
$ pip install kaggle
```

Now we are able to download the data set from kaggle

```
$ kaggle datasets download -d scolianni/mnistasjpg
```

