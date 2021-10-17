import librosa
import numpy as np
import pickle
from keras.models import model_from_json
import tensorflow

sample_rate = 22050


def extract_features(data):
    result = np.array([])

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # Mel Spectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally
    return result


def get_features_noeffect(path):
    data, sample_rate = librosa.load(path)
    res1 = extract_features(data)
    result = np.array(res1)
    return result


def get_nd_audio_cnn(path):
    feature = get_features_noeffect(path)
    feature = [feature]
    feature = np.expand_dims(feature, axis=2)
    return feature


def get_nd_audio(path):
    feature = get_features_noeffect(path)
    feature = [feature]
    feature = np.array(feature)
    return feature


def predict_emotion(model, path):
    feature = get_nd_audio(path)
    return model.predict(feature)[0]


def predict_emotion_cnn(model_path, model, path):
    feature = get_nd_audio_cnn(path)
    encoder = pickle.load(open(model_path + '/../encoder.pkl', 'rb'))
    return encoder.inverse_transform(model.predict(feature))[0][0]


def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model


def load_model_cnn(model_path):
    json_file = open(model_path + '/model_json.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_path + "/cnn.h5")
    opt = tensorflow.keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return loaded_model
