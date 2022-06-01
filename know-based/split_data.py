import numpy as np
import random
import json
import os


def split_data(X, idxTrain, idxTest, idxMovie, train_filepath, test_filepath):
    xTrain = X[idxTrain, :]
    generate_signals(xTrain, idxMovie, train_filepath)
    xTest = X[idxTest, :]
    generate_signals(xTest, idxMovie, test_filepath)


def generate_signals(X, idx, root_folder):
    '''
        - cada señal es un vector de 10.000 floats => 40.000B => 40MB
        - Queremos ocupar 2GB de RAM por cada archivo => 2000MB
        -------------------------------------------------------------
        - 50 señales por archivo
    '''

    if not os.path.exists(f"{root_folder}/signals"):
        os.makedirs(f"{root_folder}/signals")

    if not os.path.exists(f"{root_folder}/labels"):
        os.makedirs(f"{root_folder}/labels")

    n_elem = 0
    info = {}
    signals_matrix = []
    labels_matrix = []

    for i in range(X.shape[0]):

        SAMPLES_PER_USER = 50

        ref = X[i, :]

        non_zero_elements = set(np.where(ref != 0)[0])

        index_target_ratings = set(idx).intersection(non_zero_elements)

        if len(index_target_ratings) < SAMPLES_PER_USER:
            samples = len(index_target_ratings)
        else:
            samples = SAMPLES_PER_USER

        random_sampled_index = random.sample(index_target_ratings, samples)

        for idx2 in random_sampled_index:
            ref_signal = ref.copy()
            ref_signal[idx2] = 0
            label = np.zeros(len(ref_signal))
            label[idx2] = ref[idx2]

            signals_matrix.append(ref_signal)
            labels_matrix.append(label)
            n_elem += 1

            # Si hay 50 señales appendeamos la señal
            if n_elem % 50 == 0:
                filename = int(n_elem / 50)
                with open(f"{root_folder}/signals/{filename}.npy", 'wb') as f:
                    signals_matrix = np.array(signals_matrix)
                    np.save(f, signals_matrix)

                with open(f"{root_folder}/labels/{filename}.npy", 'wb') as f:
                    labels_matrix = np.array(labels_matrix)
                    np.save(f, labels_matrix)

                info[filename] = [f"{root_folder}/signals/{filename}.npy", f"{root_folder}/labels/{filename}.npy"]

                signals_matrix = []
                labels_matrix = []

    # dump de las que quedan
    filename = int(n_elem / 50) + 1
    with open(f"{root_folder}/signals/{filename}.npy", 'wb') as f:
        signals_matrix = np.array(signals_matrix)
        np.save(f, signals_matrix)

    with open(f"{root_folder}/labels/{filename}.npy", 'wb') as f:
        labels_matrix = np.array(labels_matrix)
        np.save(f, labels_matrix)

    info[filename] = [f"{root_folder}/signals/{filename}.npy", f"{root_folder}/labels/{filename}.npy"]

    info["size"] = n_elem

    with open(f"{root_folder}/info.json", 'w') as outfile:
        json.dump(info, outfile)
