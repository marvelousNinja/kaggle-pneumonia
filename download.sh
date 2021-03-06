#!/bin/bash
set -e
set -v

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle
cd ./data

kaggle competitions download -w -c rsna-pneumonia-detection-challenge

unzip stage_1_test_images.zip -d ./test
rm stage_1_test_images.zip
unzip stage_1_train_images.zip -d ./train
rm stage_1_train_images.zip
unzip stage_1_detailed_class_info.csv.zip
chmod 600 stage_1_detailed_class_info.csv
rm stage_1_detailed_class_info.csv.zip
unzip stage_1_train_labels.csv.zip
chmod 600 stage_1_train_labels.csv
rm stage_1_train_labels.csv.zip
