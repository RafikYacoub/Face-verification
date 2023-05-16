Instructions to get the notebook to work correctly:

1- !pip install kaggle
2- get the json token file from kaggle to be able to download the dataset (follow this tutorial: https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/)
3- change the path to any google drive you would like to use.
4- load the images into colab session that are in the zip folder named "verification images".

list of dependencies to  be installed and imported:
import os
import cv2
import ast
import time
import random
import numpy as np
import pandas as pd

#for the pre-trained model
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import Xception,resnet50
from tensorflow.keras.models import Model, Sequential

#for plotting
import seaborn as sns
import matplotlib.pyplot as plt
tf.__version__, np.__version__



#for data augmentation
from skimage.transform import rotate, rescale
from skimage.exposure import adjust_gamma
from skimage.util import random_noise
from skimage.filters import gaussian
from PIL import Image, ImageEnhance

#for evaluation metrics
from tensorflow.keras import backend, layers, metrics
from tensorflow.keras.utils import plot_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, log_loss

#for face detection and alignment
from numpy import loadtxt
!pip install deepface
from deepface import DeepFace
from deepface.commons import functions
from PIL import Image

#for the live demo
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import PIL
from PIL import *
import io
import html
import time


#for google drive access
from google.colab import drive
drive.mount('/content/drive')