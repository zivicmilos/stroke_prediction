import pickle
import random

import numpy as np

from ann_comp_graph import NeuralNetwork, NeuralLayer
from matplotlib import pyplot, pyplot as plt
# from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Dense


def load_data(path):
    analysis = []
    inputs = []
    output = []
    notNA = []
    with open(path, 'r') as file:
        for i, data in enumerate(file):
            if i > 0:
                row = data.strip().split(',')
                if row[9] != 'N/A':
                    notNA.append(float(row[9]))

    mean_value = np.mean(notNA)
    with open(path, 'r') as file:
        for i, data in enumerate(file):
            if i > 0:
                row = data.strip().split(',')
                analysis.append([row[1], row[5], row[6], row[7], row[10], float(row[11])])
                  # Svi atributi
                inputs.append(process_data(row[1]) + [float(row[2])] + [float(row[3])] + [float(row[4])] +
                              process_data(row[5]) + process_data(row[6]) + process_data(row[7]) + [float(row[8])] +
                              [float(row[9]) if row[9] != 'N/A' else mean_value] + process_data(row[10]))
                """  # Izbaceni neki atributi
                inputs.append([float(row[2])] + [float(row[3])] + [float(row[4])] + process_data(row[5]) +
                              process_data(row[6]) + [float(row[8])] + [float(row[9]) if row[9] != 'N/A' else mean_value]
                              + process_data(row[10]))"""
                output.append([float(row[11])])
    normalize(inputs)
    return inputs, output, analysis


def normalize(x):
    dim = len(x[0])
    m = np.mean(x)
    d = np.std(x)
    for i in range(len(x)):
        for j in range(dim):
            if x[i][j] not in (1, 0):
                x[i][j] = (x[i][j] - m) / d


def process_data(s):
    if s == 'Male':
        return [1., 0., 0.]
    elif s == 'Female':
        return [0., 1., 0.]
    elif s == 'Other':
        return [0., 0., 1.]
    elif s == 'Other':
        return [0., 0., 1.]
    elif s == 'Yes':
        return [1.]
    elif s == 'No':
        return [0.]
    elif s == 'Self-employed':
        return [1., 0., 0., 0., 0.]
    elif s == 'Private':
        return [0., 1., 0., 0., 0.]
    elif s == 'Govt_job':
        return [0., 0., 1., 0., 0.]
    elif s == 'Never_worked':
        return [0., 0., 0., 1., 0.]
    elif s == 'children':
        return [0., 0., 0., 0., 1.]
    elif s == 'never smoked':
        return [1., 0., 0., 0.]
    elif s == 'formerly smoked':
        return [0., 1., 0., 0.]
    elif s == 'smokes':
        return [0., 0., 1., 0.]
    elif s == 'Unknown':
        return [0., 0., 0., 1.]
    elif s == 'Urban':
        return [1.]
    elif s == 'Rural':
        return [0.]


def analise(for_analysis):
    plt.style.use('ggplot')

    x = ['male', 'female', 'married', 'never married', 'self employed', 'private', 'govt job', 'never worked',
         'children', 'urban', 'rural', 'never smoked', 'former smoker', 'smokes', 'unknown']
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for a in for_analysis:
        if a[0] == 'Male' and a[5] == 1:
            y[0] += 1
        elif a[0] == 'Female' and a[5] == 1:
            y[1] += 1

        if a[1] == 'Yes' and a[5] == 1:
            y[2] += 1
        elif a[1] == 'No' and a[5] == 1:
            y[3] += 1

        if a[2] == 'Self-employed' and a[5] == 1:
            y[4] += 1
        elif a[2] == 'Private' and a[5] == 1:
            y[5] += 1
        elif a[2] == 'Govt_job' and a[5] == 1:
            y[6] += 1
        elif a[2] == 'Never_worked' and a[5] == 1:
            y[7] += 1
        elif a[2] == 'children' and a[5] == 1:
            y[8] += 1

        if a[3] == 'Urban' and a[5] == 1:
            y[9] += 1
        elif a[3] == 'Rural' and a[5] == 1:
            y[10] += 1

        if a[4] == 'never smoked' and a[5] == 1:
            y[11] += 1
        elif a[4] == 'formerly smoked' and a[5] == 1:
            y[12] += 1
        elif a[4] == 'smokes' and a[5] == 1:
            y[13] += 1
        elif a[4] == 'Unknown' and a[5] == 1:
            y[14] += 1
        if a[5] == 0:
            break
    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, y, color='green')
    plt.xlabel("Uzroci")
    plt.ylabel("Broj srcanih udara")
    plt.title("Broj srcanih udara")

    plt.xticks(x_pos, x)

    plt.show()


def oversample(X, y, output):
    while len(X) > len(y):
        y.append(y[round(random.uniform(0, 248))].copy())
        output.insert(0, [1.0])


if __name__ == '__main__':
    inputs, output, for_analysis = load_data('./../data/dataset.csv')
    y = inputs[:249]
    X = inputs[249:]
    oversample(X, y, output)
    inputs = y + X

    train_inputs = []
    test_inputs = []
    train_output = []
    test_output = []
    for idx, data in enumerate(inputs):
        if idx < 3403:
            train_inputs.append(data)
        elif idx < 4861:
            test_inputs.append(data)
        elif idx < 8264:
            train_inputs.append(data)
        else:
            test_inputs.append(data)

    for idx, data in enumerate(output):
        if idx < 3403:
            train_output.append(data)
        elif idx < 4861:
            test_output.append(data)
        elif idx < 8264:
            train_output.append(data)
        else:
            test_output.append(data)

    analise(for_analysis)

    """nn = NeuralNetwork()  # Treniranje skolske mreze
    nn.add(NeuralLayer(15, 8, 'tanh'))
    nn.add(NeuralLayer(8, 5, 'tanh'))
    nn.add(NeuralLayer(5, 1, 'sigmoid'))

    history = nn.fit(train_inputs, train_output, learning_rate=0.01, momentum=0.9, nb_epochs=20
                     , shuffle=True, verbose=1)

    with open('trainedSchool', 'wb') as file:
        pickle.dump(nn, file)

    pyplot.plot(history)
    pyplot.show()

    with open('trainedSchool', 'rb') as file:
        nn = pickle.load(file)

    correct_ = 0
    all_ = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(test_inputs)):
        all_ += 1
        if round(nn.predict(test_inputs[i])[0]) == test_output[i][0]:
            correct_ += 1
            if test_output[i][0] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if test_output[i][0] == 1:
                fn += 1
            else:
                fp += 1
        print(tp, tn, fp, fn, sep=' ')

        print('Accuracy: {:0.2f}%'.format(correct_ * 100 / all_))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = precision * recall / (precision + recall)

    print('\nFinal accuracy: {:0.2f}%'.format(correct_ * 100 / all_))
    print('Precision: {:0.2f}%'.format(precision * 100))
    print('Recall: {:0.2f}%'.format(recall * 100))
    print('F1 score: {:0.2f}%'.format(f1 * 100))"""

    """nn = Sequential()  # Keras, za proveru
    nn.add(Dense(12, input_dim=19, activation='tanh'))
    nn.add(Dense(8, activation='tanh'))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    nn.fit(np.array(train_inputs), np.array(train_output), epochs=150)

    with open('trained3', 'wb') as file:
        pickle.dump(nn, file)"""

    with open('trained3', 'rb') as file:
        nn = pickle.load(file)

    correct_ = 0
    all_ = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(test_inputs)):
        all_ += 1
        if round(nn.predict(np.array([test_inputs[i], ]))[0][0]) == test_output[i][0]:
            correct_ += 1
            if test_output[i][0] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if test_output[i][0] == 1:
                fn += 1
            else:
                fp += 1
        print(tp, tn, fp, fn, sep=' ')

        print('Accuracy: {:0.2f}%'.format(correct_*100/all_))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = precision * recall / (precision + recall)

    print('\nFinal accuracy: {:0.2f}%'.format(correct_*100/all_))
    print('Precision: {:0.2f}%'.format(precision * 100))
    print('Recall: {:0.2f}%'.format(recall * 100))
    print('F1 score: {:0.2f}%'.format(f1 * 100))
