# Speech-Emotion-Recognition
A Web app which is predicting emotion of a uploaded .wav file

## TO USE
### First pull the project from Docker Hub
https://hub.docker.com/repository/docker/sercaksoy/speech-emotion-recognition-streamlit

### Then find out images Id
<img src="https://github.com/sercaksoy/Speech-Emotion-Recognition/blob/main/ss/3.jpeg">

### Run the image with the command
```
docker run -p 8501:8501 <image-id>
```
### Open it
Go to  http://localhost:8501 in your browser
<img src="https://github.com/sercaksoy/Speech-Emotion-Recognition/blob/main/ss/1.jpeg">

### Drag and drop the audio file
It will automatically predict based on the variation you choose from the left bar.
<img src="https://github.com/sercaksoy/Speech-Emotion-Recognition/blob/main/ss/2.jpeg">

## REQUIREMENTS
- streamlit==0.88.0
- pandas==1.3.2
- pydub==0.25.1
- librosa==0.8.1
- numpy==1.19.5
- pickleshare==0.7.5
- keras==2.6.0
- tensorflow==2.6.0

## TRAINING MODELS
SER-ALL notebook used to extract features from audios and training part. Notebook already contains explanations with comment lines. Datasets may downloaded from the following links.

### DATASETS
Crema-d : https://www.kaggle.com/ejlok1/cremad </br>
Ravdess : https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio </br>
Savee   : https://www.kaggle.com/ejlok1/surrey-audiovisual-expressed-emotion-savee </br>
Tess    : https://www.kaggle.com/ejlok1/toronto-emotional-speech-set-tess </br>
</br>
tess-ravdess : Tess and Ravdess datasets used together. </br>
merged4      : All 4 datasets used together. </br>

## MODEL1,MODEL2 AND MODEL6
**Model 1 :** I just extracted features and trained the models with the data. </br>
**Model 2 :** After splitting train and test part of the data while extracting features I added noise and expanded the training dataset x3. </br>
**Model 3 :** To reduce soft and more complex emotions from the dataset I selected only auidos with labeled neutral, happy, sad, angry and trained with augmentation. </br>
