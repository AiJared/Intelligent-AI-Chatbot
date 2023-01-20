import random
import numpy as np
import json
import pickle

import nltk
from nltk.stem import WordNetLemmatizer

import tensorflow

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD