import streamlit as st
import audio_func
import pandas as pd
from pydub import AudioSegment

#models_path = "C:/Users/Administrator/Documents/sss/models"
models_path = "models"

st.title('Speech Emotion Recognition')

dataset_name = st.sidebar.selectbox("Select Dataset", ("SAVEE", "RAVDESS", "TESS", "CREMA", "TESS-RAVDESS", "MERGED-4"))
creation_type = st.sidebar.selectbox("Data Mode", ("Normal", "Data Augmentation", "Data Augmentation(4CLASS)"))
model_name = st.sidebar.selectbox("Select Classifier", ("Gradient Boosting", "Decision Tree", "K Neighbors",
                                                        "Logistic Regression", "Random Forest", "1D CNN"))

df = pd.DataFrame()
df_2 = pd.DataFrame()

if dataset_name == "SAVEE":
    df = pd.DataFrame([(400, 0, 400)],
                      columns=['Total Audio Count', 'Women Audio Count', 'Men Audio Count'])
    df_2 = pd.DataFrame(['angry', 'surprise', 'neutral', 'disgust', 'sad', 'fear', 'happy'], columns=['Emotions'])
elif dataset_name == "RAVDESS":
    df = pd.DataFrame([(1061, 525, 536)],
                      columns=['Total Audio Count', 'Women Audio Count', 'Men Audio Count'])
    df_2 = pd.DataFrame(['angry', 'surprise', 'neutral', 'disgust', 'sad', 'fear', 'happy'], columns=['Emotions'])
elif dataset_name == "TESS":
    df = pd.DataFrame([(2380, 2380, 0)],
                      columns=['Total Audio Count', 'Women Audio Count', 'Men Audio Count'])
    df_2 = pd.DataFrame(['angry', 'surprise', 'neutral', 'disgust', 'sad', 'fear', 'happy'], columns=['Emotions'])
elif dataset_name == "CREMA":
    df = pd.DataFrame([(6326, 3342, 2984)],
                      columns=['Total Audio Count', 'Women Audio Count', 'Men Audio Count'])
    df_2 = pd.DataFrame(['angry', 'neutral', 'disgust', 'sad', 'fear', 'happy'], columns=['Emotions'])
elif dataset_name == "TESS-RAVDESS":
    df = pd.DataFrame([(3441, 2905, 536)],
                      columns=['Total Audio Count', 'Women Audio Count', 'Men Audio Count'])
    df_2 = pd.DataFrame(['angry', 'surprise', 'neutral', 'disgust', 'sad', 'fear', 'happy'], columns=['Emotions'])
else:
    df = pd.DataFrame([(10175, 5889, 4286)],
                      columns=['Total Audio Count', 'Women Audio Count', 'Men Audio Count'])
    df_2 = pd.DataFrame(['angry', 'surprise', 'neutral', 'disgust', 'sad', 'fear', 'happy'], columns=['Emotions'])


def bring_model(dataset_name, creation_type, model_name):
    if dataset_name == "SAVEE":
        model_path = models_path + "/savee"
    elif dataset_name == "RAVDESS":
        model_path = models_path + "/ravdess"
    elif dataset_name == "TESS":
        model_path = models_path + "/tess"
    elif dataset_name == "CREMA":
        model_path = models_path + "/crema"
    elif dataset_name == "TESS-RAVDESS":
        model_path = models_path + "/tess-ravdess"
    else:
        model_path = models_path + "/merged4"

    model_path = model_path + "/model"

    if creation_type == "Normal":
        model_path = model_path + '1'
    elif creation_type == "Data Augmentation":
        model_path = model_path + '2'
    else:
        model_path = model_path + '6'

    if model_name == "Gradient Boosting":
        model_path = model_path + "/gradient_b_model.pkl"
    elif model_name == "Decision Tree":
        model_path = model_path + "/dec_t_model.pkl"
    elif model_name == "K Neighbors":
        model_path = model_path + "/knn_model.pkl"
    elif model_name == "Logistic Regression":
        model_path = model_path + "/logistic_reg_model.pkl"
    elif model_name == "Random Forest":
        model_path = model_path + "/random_forest_model.pkl"
    else:
        model_path = model_path + "/cnn"

    return model_path


result_file = st.file_uploader(label=".wav File Uplaoder")

try:
    audio_file = AudioSegment.from_wav(result_file)
    audio_file.export(result_file.name, format='wav')
    st.audio(open(result_file.name, 'rb').read(), format='audio/wav')
    if model_name != "1D CNN":
        st.text("Prediction : " + audio_func.predict_emotion(audio_func.load_model
                                                             (bring_model(dataset_name, creation_type, model_name)),
                                                             result_file.name))
    else:
        model_path_m = bring_model(dataset_name, creation_type, model_name)
        emotion = audio_func.predict_emotion_cnn(model_path_m,
                                                 audio_func.load_model_cnn(model_path_m), result_file.name)
        st.text("Prediction : " + emotion)
except:
    st.text('Upload valid .wav file')

st.title('About ' + dataset_name + ' Training Dataset')

st.dataframe(df)
st.dataframe(df_2)
