import librosa
import librosa.display

from matplotlib import pyplot as plt

audio_path = 'elec301-2021-music-genres/data/data/blues1.wav'
x , sr = librosa.load(audio_path)

print(type(x), type(sr))



mfccs = librosa.feature.mfcc(x, sr=sr)
print(mfccs.shape)#Displaying  the MFCCs:

#fig = plt.figure()
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.show()
