from google.colab import drive
drive.mount('/content/drive')

import os
Root = "/content/drive/MyDrive/Colab Notebooks/RAVDESS_Emotional_speech_audio"
os.chdir(Root)

import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import librosa.display

#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

# Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust']

#Load the data and extract features for each sound file
def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("/content/drive/MyDrive/Colab Notebooks/RAVDESS_Emotional_speech_audio/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

import IPython.display as ipd
from IPython.display import Image

data, sampling_rate = librosa.load("/content/drive/MyDrive/Colab Notebooks/RAVDESS_Emotional_speech_audio/Actor_01/03-01-01-01-01-01-01.wav")
# To play audio this in the jupyter notebook
ipd.Audio('/content/drive/MyDrive/Colab Notebooks/RAVDESS_Emotional_speech_audio/Actor_01/03-01-01-01-01-01-01.wav')

"""plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)
"""

#Split the dataset
x_train,x_test,y_train,y_test=load_data(test_size=0.25)

x_train

#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

#Train the model
model.fit(x_train,y_train)

#Predict for the test set
y_pred=model.predict(x_test)

y_pred

#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))

mylabel= "Accuracy"
b= [accuracy, 1- accuracy]
plt.pie(b)
plt.show()

from sklearn.metrics import accuracy_score, f1_score

a= f1_score(y_test, y_pred,average=None)

s = np.array(a)
fig, axis = plt.subplots(figsize =(10, 5))
axis.hist(s, align='mid', color='yellow', edgecolor='black', linewidth=2)
plt.xlabel('F1 Scores')
plt.ylabel('Emotions')
plt.title('Distribution of Prediction Scores')
plt.show()

import pandas as pd
df= pd.DataFrame({'Emotions Observed': observed_emotions, 'F1 Score': a})
df.head()

import pandas as pd
df=pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})
df.head(100)

import pickle
# Writing different model files to file
with open( 'modelForPrediction1.sav', 'wb') as f:
    pickle.dump(model,f)

filename = 'modelForPrediction1.sav'
loaded_model = pickle.load(open(filename, 'rb')) # loading the model file from the storage

feature=extract_feature("/content/drive/MyDrive/Colab Notebooks/RAVDESS_Emotional_speech_audio/Actor_01/03-01-02-01-01-02-01.wav", mfcc=True, chroma=True, mel=True)

feature=feature.reshape(1,-1)

prediction=loaded_model.predict(feature)
prediction
