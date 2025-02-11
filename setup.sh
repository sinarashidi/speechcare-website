#!/bin/bash

apt update && apt install zip ffmpeg -y

pip install gdown

pip install -r requirements.txt

echo "Downloading Dataset and Model file..."
gdown 1sKpjxsSdy27CR8LZ6mq-8RF_HG_Kphzw # Dataset file
gdown 1Cixa_7eunClHa35fWF6vxrkmwb87IZhz # Model checkpoint file

echo "Extracting the dataset..."
unzip -q NIA_Data-denoise.zip -d ./Datasets/