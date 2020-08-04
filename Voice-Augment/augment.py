import numpy as np
import librosa
import os
import resemblyzer
import pickle
def augment_data():
    noise_names = []
    for noise_path in os.listdir('./noise'):
        noise_name = noise_path.split('.')[0]
        noise = f'./noise/{noise_path}'
        for username in os.listdir('./data'):
            data_encode = []
            for user_file in os.listdir(f'./data/{username}'):
                y, sr = librosa.load(f'./data/{username}/{user_file}',
                                     sr=16000)
                encoded_data = resemblyzer.preprocess_wav(f'./data/{username}/{user_file}')
                data_encode.append(encoded_data)
                for i in range(1):
                    choice = i
                    print(choice)

                    if choice == 1:
                        aug = pitch(mix_bg(y, noise), sr, 0.2)
                    elif choice == 2:
                        aug = speed(mix_bg(y, noise), 1.2)
                    else:
                        aug = mix_bg(y, noise)

                    if not os.path.exists(f'./augmented_data/{username}'):
                        os.mkdir(f'./augmented_data/{username}')

                    librosa.output.write_wav(
                        f'./augmented_data/{username}/{noise_name}_{i}.wav',
                        aug, sr)
                    encoded_data = resemblyzer.preprocess_wav(f'./augmented_data/{username}/{noise_name}_{i}.wav')
                    data_encode.append(encoded_data)
                    with open(f'data/{username}_encoded_wav.pickle', 'wb') as handle:
                        pickle.dump(data_encode, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def speed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)


def mix_bg(data, noise_path):
    bg_y, br_sr = librosa.load(noise_path, sr=16000)
    bg_y = bg_y*0.5
    start_ = np.random.randint(bg_y.shape[0] - data.shape[0])
    bg_slice = bg_y[start_:start_ + data.shape[0]]
    wav_with_bg = data + bg_slice * 0.5
    return wav_with_bg


augment_data()