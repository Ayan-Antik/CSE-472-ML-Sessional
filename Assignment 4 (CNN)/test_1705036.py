import numpy as np
import cv2, os, pickle, sys
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from train_1705036 import ConvolutionLayer, PoolingLayer, ActivationLayer, FlatteningLayer, FullyConnectedLayer, SoftMaxLayer, ModelBuilder
import matplotlib.pyplot as plt
import seaborn as sns

# load data
def load_t_images(folder_path):
    images = []
    files = []
    for filename in sorted(os.listdir(folder_path)):
        # load the image using OpenCV
        img = cv2.imread(os.path.join(folder_path, filename))
        # compress image pixels
        img = cv2.resize(img, (28, 28))
        # convert the image from BGR to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # convert the image to a NumPy array of shape (channel, height, width)
        img = (255 - np.transpose(img, (2, 0, 1))) / 255
        
        images.append(img)
        files.append(filename)
        
    return images, files


def load_labels(filepath):
    labels = np.loadtxt(filepath, delimiter=',', skiprows=1, usecols=2, dtype=int)
    return labels

if __name__ == '__main__':

    path_to_images = sys.argv[1] if len(sys.argv) > 1 else 'data/training-d/'
    validation = True if input('Validation? (y/n) ') == 'y' else False
    print('Loading data...')
    images, files = load_t_images(path_to_images)
    labels = []
    if validation:
        labels = load_labels('data/training-d.csv')
    print('Data loaded.')

    # load model
    print('Loading model...')

    model = ModelBuilder().build()

    with open('1705036_model.pkl', 'rb') as f:
        model_weights_biases = pickle.load(f)

    # len(model_weights)
    # set model weights and bias
    i = 0
    for layer in model:
        if hasattr(layer, 'weights'):
            print(layer)
            layer.weights = model_weights_biases[i * 2]
            layer.biases = model_weights_biases[i * 2 + 1]
            i += 1

    print('Model loaded.')
    # predict

    images = np.array(images)
    for layer in model:
        images = layer.forward(images)
    
    print("forward pass completed")
    # get the index of the maximum value in the output
    predictions = np.argmax(images, axis=1)
    if validation:
        labels = np.array(labels)
        # calculate the accuracy, f1 score
        accuracy = accuracy_score(labels, predictions)
        print('Accuracy: {:.2f}%'.format(accuracy * 100))

        f1 = f1_score(labels, predictions, average='macro')
        print('F1 score: {:.2f}%'.format(f1 * 100))

        cm = confusion_matrix(labels, predictions)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion matrix')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()
    
    else:
        with open('1705036_prediction.csv', 'w') as f:
            f.write('FileName, Digit\n')
            for file, prediction in zip(files, predictions):
                f.write(f'{file}, {str(prediction)}\n')
