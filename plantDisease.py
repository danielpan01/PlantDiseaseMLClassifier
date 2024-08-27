from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

save_dir = 'DataPlotting'

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
        for filename in tqdm(sorted(folder_path.iterdir()), desc=f'Processing {folder_name}'):
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

def train_model(inputs_train, targets_train, learning_rate_init=0.01, hidden_layer_sizes=(100,)):
    model = MLPClassifier(
        random_state=0,
        max_iter=1000,
        verbose=1,
        learning_rate_init=learning_rate_init,
        hidden_layer_sizes=hidden_layer_sizes
    )
    model.fit(inputs_train, targets_train)
    return model

def evaluate_model(model, data_partition, inputs, targets):
    predictions = model.predict(inputs)
    accuracy = accuracy_score(targets, predictions)
    print(f'{data_partition} Accuracy: {accuracy:.4f}')
    display_confusion_matrix(targets, predictions, plot_title=f'{data_partition} Performance')

def display_confusion_matrix(target, predictions, labels=['Healthy', 'Unhealthy'], plot_title='Performance'):
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(plot_title)
    plot_path = os.path.join(save_dir, f'{plot_title.replace(" ", "_")}.png')
    plt.savefig(plot_path)
    plt.close()

def time_func(msg, func, *args):
    start_time = time.time()
    res = func(*args)
    print(f"{msg}{time.time() - start_time:.2f} seconds")
    return res

def train_and_plot_learning_rate(init_rates, inputs_train, targets_train):
    plt.figure(figsize=(12, 8))

    models = []
    for rate in init_rates:
        print(f"Training with learning_rate_init={rate}")
        model = train_model(inputs_train, targets_train, learning_rate_init=rate)
        plt.plot(model.loss_curve_, label=f'LR={rate}')
        models.append(model)

    plt.title('Loss Curve for Different Learning Rates')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'loss_curve_learning_rates.png'))
    plt.show()

    return models

def plot_best_losses(models, init_rates):
    best_losses = [model.loss_curve_[-1] for model in models]

    plt.figure(figsize=(10, 6))
    plt.plot(init_rates, best_losses, marker='o')
    plt.title('Best Loss for Different Learning Rates')
    plt.xlabel('Learning Rate')
    plt.ylabel('Best Loss')
    plt.xscale('log')
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'best_loss_learning_rates.png'))
    plt.show()

def main_learning_rate():
    inputs, targets = time_func('Data loading time took: ', load_data)
    inputs_train, inputs_test, targets_train, targets_test = time_func('Data preprocessing took: ', preprocess, inputs,
                                                                       targets)

    init_rates = [0.0001, 0.001, 0.01, 0.1]

    models = train_and_plot_learning_rate(init_rates, inputs_train, targets_train)
    plot_best_losses(models, init_rates)

def train_and_plot_hidden_layers(hidden_sizes, inputs_train, targets_train):
    plt.figure(figsize=(12, 8))

    models = []
    for hidden_size in hidden_sizes:
        print(f"Training with hidden_layer_sizes={hidden_size}")
        model = train_model(inputs_train, targets_train, hidden_layer_sizes=hidden_size)
        plt.plot(model.loss_curve_, label=f'Hidden Layers={hidden_size}')
        models.append(model)

    plt.title('Loss Curve for Different Hidden Layer Sizes')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'loss_curve_hidden_sizes.png'))
    plt.show()

    return models

def plot_best_losses_hidden(models, hidden_sizes):
    best_losses = [model.loss_curve_[-1] for model in models]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(hidden_sizes)), best_losses, marker='o')
    plt.title('Best Loss for Different Hidden Layer Sizes')
    plt.xlabel('Model Index')
    plt.ylabel('Best Loss')
    plt.xticks(range(len(hidden_sizes)), hidden_sizes, rotation=45)
    plt.grid(True)

    plt.savefig(os.path.join(save_dir, 'best_loss_hidden_sizes.png'))
    plt.show()

def main_hidden_layers():
    inputs, targets = time_func('Data loading time took: ', load_data)
    inputs_train, inputs_test, targets_train, targets_test = time_func('Data preprocessing took: ', preprocess, inputs,
                                                                       targets)

    hidden_sizes = [(50,), (100,), (50, 50), (100, 50, 25)]

    models = train_and_plot_hidden_layers(hidden_sizes, inputs_train, targets_train)
    plot_best_losses_hidden(models, hidden_sizes)

if __name__ == "__main__":
    main_learning_rate()
    main_hidden_layers()
