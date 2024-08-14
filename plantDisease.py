from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

def load_data():
    base_path = Path("PlantVillage")
    class_folders = {
        "Pepper__bell___Bacterial_spot": 1,
        "Pepper__bell___healthy": 0,
        "Potato___Early_blight": 1,
        "Potato___healthy": 0,
        "Potato___Late_blight": 1,
        "Tomato__Target_Spot": 1,
        "Tomato__Tomato_mosaic_virus": 1,
        "Tomato__Tomato_YellowLeaf__Curl_Virus": 1,
        "Tomato_Bacterial_spot": 1,
        "Tomato_Early_blight": 1,
        "Tomato_healthy": 0,
        "Tomato_Late_blight": 1,
        "Tomato_Leaf_Mold": 1,
        "Tomato_Septoria_leaf_spot": 1,
        "Tomato_Spider_mites_Two_spotted_spider_mite": 1
    }

    inputs = []
    targets = []

    image_size = (128, 128)

    for folder_name, label in class_folders.items():
        folder_path = base_path / folder_name
        for filename in sorted(folder_path.iterdir()):
            img = Image.open(filename).convert('RGB').resize(image_size)
            img = np.array(img).flatten()
            inputs.append(img)
            targets.append(label)

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

def time_func(msg, func, *args):
    start_time = time.time()
    res = func(*args)
    print(f"{msg}{time.time() - start_time:.2f} seconds")
    return res

def main():
    inputs, targets = time_func('Data loading time took: ', load_data)
    inputs_train, inputs_test, targets_train, targets_test = time_func('Data preprocessing took: ', preprocess, inputs, targets)
    model = time_func('Model training took: ', train_model, inputs_train, targets_train)
    time_func('Model evaluation took: ', evaluate_model, model, inputs_test, targets_test)

if __name__ == "__main__":
    main()
