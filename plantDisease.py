from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_data():
    path_healthy = Path("PlantVillage/Tomato_healthy")
    path_unhealthy = Path("PlantVillage/Tomato_Early_blight")

    inputs = []
    targets = []

    image_size = (128, 128)

    for filename in sorted(path_healthy.iterdir()):
        img = Image.open(filename).convert('RGB').resize(image_size)
        img = np.array(img).flatten()
        inputs.append(img)
        targets.append(0)

    for filename in sorted(path_unhealthy.iterdir()):
        img = Image.open(filename).convert('RGB').resize(image_size)
        img = np.array(img).flatten()
        inputs.append(img)
        targets.append(1)

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

def preprocess(inputs, targets):
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.10, random_state=0,
    )
    inputs_train = inputs_train / np.float32(255.)
    inputs_test = inputs_test / np.float32(255.)
    return inputs_train, inputs_test, targets_train, targets_test

def train_model(inputs_train, targets_train):
    model = MLPClassifier(random_state=0, max_iter=1000)
    model.fit(inputs_train, targets_train)
    return model

def evaluate_model(model, inputs_test, targets_test):
    predictions = model.predict(inputs_test)
    accuracy = accuracy_score(targets_test, predictions)
    print(f'Accuracy: {accuracy:.4f}')
    display_confusion_matrix(targets_test, predictions)

def display_confusion_matrix(target, predictions, labels=['Healthy', 'Unhealthy'], plot_title='Performance'):
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(plot_title)
    plt.show()

inputs, targets = load_data()
inputs_train, inputs_test, targets_train, targets_test = preprocess(inputs, targets)
model = train_model(inputs_train, targets_train)
evaluate_model(model, inputs_test, targets_test)
