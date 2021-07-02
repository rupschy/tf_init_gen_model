import os

import numpy as np

from vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 256
EPOCHS = 10  # init is 20

SPECTROGRAMS_PATH = "/Users/johnrupsch/Models/tf_init_model/Spectrograms"

def load_esc50(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)  # (n_bins, n_frames)
            x_train.append(spectrogram)

    x_train = np.array(x_train)
    x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(128, 1723, 1),
        conv_filters=(4, 8, 8, 8),
        conv_kernels=(1, 1, 1, 1),
        conv_strides=(1, 1, 1, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = load_esc50(SPECTROGRAMS_PATH)
    autoencoder = train(x_train[:50], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")