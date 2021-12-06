# libraries
import gc
import os
import sys
import urllib.request
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import torch
# custom libraries
sys.path.append('models')
from models.model import MobNetSimpsons


# download with progress bar
mybar = None

def show_progress(block_num, block_size, total_size):
    global mybar
    if mybar is None:
        mybar = st.progress(0.0)
    downloaded = block_num * block_size / total_size
    if downloaded <= 1.0:
        mybar.progress(downloaded)
    else:
        mybar.progress(1.0)


# CONFIG

# CONFIG

# page config
st.set_page_config(page_title="Classifiy Springfield Character",
                   page_icon="🧐",
                   layout="centered",
                   initial_sidebar_state="collapsed",
                   menu_items=None)


# HEADER

# title
st.title("Which character from the Simpsons is in the picture?")

# image cover
cover_image = Image.open(requests.get(
    "https://images5.alphacoders.com/108/1085900.jpg", stream=True).raw)
st.image(cover_image)

# description
st.write(
    "This app uses deep learning to estimate a CV Image classification task of Springfield character. [Research](https://www.kaggle.com/dailysergey/simpsons-image-classification-task) can be found in my [Kaggle profile](https://www.kaggle.com/dailysergey).")


# PARAMETERS

# header
st.header('Score')

# photo upload
your_image = st.file_uploader("1. Upload image of Springfield character")
if your_image is not None:

    # check image format
    image_path = 'tmp/' + your_image.name
    if ('.jpg' not in image_path) and ('.JPG' not in image_path) and ('.jpeg' not in image_path) and ('.bmp' not in image_path):
        st.error(
            'Please upload .jpeg, .jpg or .bmp file. Use english or number namings')
    else:

        # save image to folder
        with open(image_path, "wb") as f:
            f.write(your_image.getbuffer())

        # display image
        st.success('Photo uploaded.')

# model selection
model_name = st.selectbox(
    '3. Choose a model for scoring your Springfield Character.',
    ['mobNetv3small', 'VGG16', 'mobNetLarge'])


# MODELING

# compute pawpularity
if st.button('Compute prediction'):

    # check if image is uploaded
    if your_image is None:
        st.error('Please upload an image first.')
    else:
        # specify paths
        if model_name == 'mobNetv3small':
            weight_path = 'https://github.com/dailysergey/streamlit-simpsons/releases/download/models/mobNetv3small.pth'
            model_path = 'models/mobNetv3small/'
        elif model_name == 'VGG16':
            weight_path = 'https://github.com/dailysergey/streamlit-simpsons/releases/download/models/vgg_16.pth'
            model_path = 'models/vgg16/'
        elif model_name == 'mobNetLarge':
            weight_path = 'https://github.com/dailysergey/streamlit-simpsons/releases/download/models/mobNetLarge.pth'
            model_path = 'models/mobNetLarge/'

        # download model weights
        if not os.path.isfile(model_path + 'pytorch_model.pth'):
            with st.spinner('Downloading model weights. This is done once and can take a minute...'):
                resp = None
                try:
                    print('Model is loading')
                    resp = urllib.request.urlretrieve(weight_path, model_path + 'pytorch_model.pth', show_progress)
                    st.success(f'{model_name} is loaded')
                except Exception as e:
                    st.error(f'{model_name} is Not loaded. Error Text: {e}, Resp: {resp}')

        # compute predictions
        with st.spinner('Computing prediction...'):

            # clear memory
            gc.collect()

            # initialize model
            model = MobNetSimpsons(model_name=model_name)

            # predict
            pred = model.predict(image_path)

            # process image
            try:
                your_image = cv2.imread(image_path)
                your_image = cv2.cvtColor(your_image, cv2.COLOR_BGR2RGB)
                # display results
                col1, col2 = st.columns(2)
                col1.image(cv2.resize(your_image, (256, 256)))
                col2.metric('It\'s ', pred[0])
                col2.metric('Percentile',  str(round(pred[1], 2)) + '%')
                col2.write('**Note:** Prediction ranges from 0 to 100.')

                # clear memory
                del model
                gc.collect()

                # remove image from tmp folder
                #os.remove(image_path)

                # celebrate
                st.success('Well done! Thanks for scoring your character :)')
            except:
                if your_image is None:
                    st.error('Upload image with english or number namings')


# CONTACT

# header
st.header("Contact")

# website link
st.write("Check out [my website](https://gusevski.com).")

# profile links
st.write("[![Twitter](https://img.shields.io/badge/-Twitter-4B9AE5?style=flat&logo=Twitter&logoColor=white&link=https://www.twitter.com/daily_sergey)](https://www.twitter.com/daily_sergey) [![Kaggle](https://img.shields.io/badge/-Kaggle-5DB0DB?style=flat&logo=Kaggle&logoColor=white&link=https://www.kaggle.com/dailysergey)](https://www.kaggle.com/dailysergey) [![GitHub](https://img.shields.io/badge/-GitHub-2F2F2F?style=flat&logo=github&logoColor=white&link=https://www.github.com/dailysergey)](https://www.github.com/dailysergey)  [![Tea](https://img.shields.io/badge/-Buy_me_a_coffee-yellow?style=flat&logo=buymeacoffee&logoColor=white&link=https://www.buymeacoffee.com/gusevski)](https://www.buymeacoffee.com/gusevski)")

# copyright
st.text("© 2021 Sergey Guskov")
