import hashlib
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import pipeline
import torch

save_dir = Path('DataPlotting')
save_dir.mkdir(exist_ok=True)

def get_hashed_filename(class_folders):
    class_string = "--".join(f"{k}-{v}" for k, v in class_folders.items())
    return hashlib.md5(class_string.encode()).hexdigest()

def load_data(return_array=True):
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

    hashed_filename = get_hashed_filename(class_folders)
    saved_imgs_path = Path(f'saved_imgs_{hashed_filename}.npz')

    if not saved_imgs_path.exists() or not return_array:
        inputs = []
        targets = []
        for folder_name, label in class_folders.items():
            folder_path = base_path / folder_name
            filenames = sorted(p for p in folder_path.iterdir() if p.suffix == '.JPG')
            for filename in filenames:
                inputs.append(filename)
                targets.append(label)

        if return_array:
            image_size = (128, 128)
            processed_inputs = []
            for img_path in tqdm(inputs, desc='Extracting pixels from images'):
                img = Image.open(img_path).convert('RGB').resize(image_size)
                img = np.array(img).flatten()
                processed_inputs.append(img)
            inputs = np.array(processed_inputs)
            targets = np.array(targets)
            np.savez_compressed(saved_imgs_path, inputs=inputs, targets=targets)
            print(f'Successfully saved "{str(saved_imgs_path)}"')
        else:
            targets = np.array(targets)

    else:
        arr = np.load(saved_imgs_path)
        inputs = arr['inputs']
        targets = arr['targets']
        print(f'Loaded "{str(saved_imgs_path)}"')

    return inputs, targets

def preprocess(inputs, targets):
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.10, random_state=0,
    )
    inputs_train = inputs_train / np.float32(255.)
    inputs_test = inputs_test / np.float32(255.)
    return inputs_train, inputs_test, targets_train, targets_test

def train_model(inputs_train, targets_train, learning_rate_init=0.001, hidden_layer_sizes=(100,)):
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
    diff_hyperparams = "  ".join((
        f'{k + ": " + str(v):^35}'
        for k, v in sorted(
        set(model.get_params().items()) - set(MLPClassifier().get_params().items())
    )
    ))
    print(
        f'{data_partition:5s} Accuracy: {accuracy * 100:.3f}%  '
        f'{diff_hyperparams}'
    )
    return predictions

def plot_all(models, inputs_train, targets_train, inputs_test, targets_test, argname, argvals):
    plt.figure(figsize=(12, 8))

    # Plot loss curves
    for model, argval in zip(models, argvals):
        plt.plot(model.loss_curve_, label=f'{argname}={argval}')
    plt.title(f'Loss Curve for Different {argname}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'loss_curve_{argname}.png')
    plt.close()

    # Plot best losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(argvals)), [model.best_loss_ for model in models], marker='o')
    plt.title(f'Best Loss for Different {argname}')
    plt.xlabel(argname)
    plt.ylabel('Best Loss')
    plt.xticks(range(len(argvals)), argvals, rotation=45)
    if argname == 'learning_rate_init':
        plt.xscale('log')
    plt.grid(True)
    plt.savefig(save_dir / f'best_loss_{argname}.png')
    plt.close()

    # Plot confusion matrices
    for model in models:
        for data_partition, inputs, targets in [
            ('Train', inputs_train, targets_train),
            ('Test', inputs_test, targets_test)
        ]:
            predictions = evaluate_model(model, data_partition, inputs, targets)
            cm = confusion_matrix(targets, predictions)
            cm_display = ConfusionMatrixDisplay(cm, display_labels=['Healthy', 'Unhealthy'])
            fig, ax = plt.subplots()
            cm_display.plot(ax=ax)
            ax.set_title(f'{data_partition} Confusion Matrix ({argname}={model.get_params()[argname]})')
            plot_path = save_dir / f'{data_partition.lower()}_confusion_matrix_{argname}_{model.get_params()[argname]}.png'
            plt.savefig(plot_path)
            plt.close()

def deep_model_classify(img_paths, targets, batch_size=30):
    dev = torch.device('cuda') if torch.cuda.is_available() else (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
    model = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch16", device=dev)
    predictions = []
    num_batches = len(img_paths) // batch_size + (1 if len(img_paths) % batch_size != 0 else 0)

    for i in tqdm(range(num_batches), desc="Classifying images"):
        batch_paths = [str(path) for path in img_paths[i * batch_size:(i + 1) * batch_size]]
        batch_results = model(batch_paths, candidate_labels=['healthy tomato plant', 'unhealthy tomato plant'])

        for result in batch_results:
            label = result[0]['label']
            predictions.append(label)

    return predictions

def deep_model_accuracy(predictions, targets):
    correct = 0
    for prediction, target in zip(predictions, targets):
        target_label = 'healthy tomato plant' if target == 0 else 'unhealthy tomato plant'
        if prediction == target_label:
            correct += 1

    accuracy = correct / len(targets)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

def shallow_model(argname, argvals):
    inputs, targets = load_data(return_array=True)
    inputs_train, inputs_test, targets_train, targets_test = preprocess(inputs, targets)
    models = [train_model(inputs_train, targets_train, **{argname: argval}) for argval in argvals]
    plot_all(models, inputs_train, targets_train, inputs_test, targets_test, argname, argvals)

def deep_model():
    img_paths, targets = load_data(return_array=False)
    predictions = deep_model_classify(img_paths, targets, batch_size=30)
    deep_model_accuracy(predictions, targets)

if __name__ == "__main__":
    shallow_model('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50, 25)])
    shallow_model('learning_rate_init', [0.0001, 0.001, 0.01, 0.1])
    #deep_model()
