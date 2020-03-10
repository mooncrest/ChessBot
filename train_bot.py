import numpy as np
import math
from scipy import sparse, stats
from sklearn.ensemble import RandomForestClassifier
import os
import pickle

DATA_DIRECTORY = './dataParser/parsedData/'
SHUFFLED_DATA = 'shuffledGames.txt'
NON_SHUFFLED_DATA = 'games.txt'

DATA_LENGTH = 1212827

def load_data(data_file, proportion, file_index=0):
    """ the <proportion> fo the data to load set to load for batch training 0 to 1 inclusive
    the <file_index> from file.tell when we last left off.
    """
    data = []
    labels = []

    lines_to_read = math.ceil(DATA_LENGTH * proportion)

    file = open(data_file, 'r')
    file.seek(file_index)

    while lines_to_read > 0:
        line = file.readline().strip()
        if (line == ''):
            break
        label, vector = line.split(",")
        data.extend((int(i) for i in vector.strip()))
        labels.append(int(label))
        lines_to_read -= 1


    new_file_index = file.tell()
    file.close()
    length = len(data) // 768

    data = np.array(data)
    data = data.reshape(length, 768)
    data = sparse.csr_matrix(data).astype(np.float64)
    labels = np.array(labels, dtype=np.float64)
    labels.reshape(length, 1)

    return data, labels, new_file_index

def create_models(data, labels):
    clf = RandomForestClassifier(max_depth=50, n_jobs=5, n_estimators=20)
    clf.fit(data, labels)
    return clf

def predict(models, data):
    model_predictions = []
    for model in models:
        predictions = model.predict(data)
        predictions = predictions.reshape(1, len(predictions))
        model_predictions.append(predictions)

    return stats.mode(np.concatenate(model_predictions, axis=1))

def test(models, data, labels):
    predictions = predict(models, data)
    count = 0
    for prediction, label in zip(predictions[0][0], labels):
        count += (prediction == label)
    return count / len(labels)

def train_best_model(data_path, batches):
    # set p as 0.000004 for 4 elements
    batch_prop = 1 / (batches + 1)
    file_index = 0
    models = []
    for i in range(batches):
        print(f"loading batch {i + 1}")
        data, labels, file_index = load_data(data_path, batch_prop, file_index)
        print(f"fitting model {i + 1}")
        models.append(create_models(data, labels))
        print(f"finished model {i + 1}")


    print("saving models")
    for i in range(batches):
        file = open(f"RandomTree{i + 1}.sav", 'wb')
        pickle.dump(models[i], file)
        file.close()

    return file_index

def load_and_test(data_path, batches, file_index):
    print("loading models")
    models = []
    for file in os.listdir("."):
        if file.endswith(".sav"):
            file = open(file, 'rb')
            models.append(pickle.load(file))

    print("testing models")
    data, labels, file_index = load_data(data_path, 1 / (batches + 1), file_index)
    accuracy = test(models, data, labels)
    print(f"model classified correctly with accuracy of: {accuracy}%")

if __name__ == '__main__':
    # set the data type you wanted

    data = DATA_DIRECTORY + SHUFFLED_DATA
    batches = 5
    file_index = train_best_model(data, batches)
    print(file_index)
    load_and_test(data, batches, file_index)

