from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

save_dir = Path('DataPlotting')
save_dir.mkdir(exist_ok=True)

def load_data():
    base_path = Path("PlantVillage")
    class_folders = {
        "Pepper__bell___Bacterial_spot": 1,
        "Pepper__bell___healthy": 0,
        # "Potato___Early_blight": 1,
        # "Potato___healthy": 0,
        # "Potato___Late_blight": 1,
        # "Tomato__Target_Spot": 1,
        # "Tomato__Tomato_mosaic_virus": 1,
        # "Tomato__Tomato_YellowLeaf__Curl_Virus": 1,
        # "Tomato_Bacterial_spot": 1,
        # "Tomato_Early_blight": 1,
        # "Tomato_healthy": 0,
        # "Tomato_Late_blight": 1,
        # "Tomato_Leaf_Mold": 1,
        # "Tomato_Septoria_leaf_spot": 1,
        # "Tomato_Spider_mites_Two_spotted_spider_mite": 1
    }

    saved_imgs_path = Path(
        f'saved_imgs_{"--".join(f"{k}-{v}" for k, v in class_folders.items())}.npz'
    )
    if not saved_imgs_path.exists():
        inputs = []
        targets = []

        image_size = (128, 128)

        for folder_name, label in class_folders.items():
            folder_path = base_path / folder_name
            filenames = sorted(p for p in folder_path.iterdir() if p.suffix == '.JPG')
            for filename in tqdm(filenames, desc=f'Processing {folder_name}'):
                img = Image.open(filename).convert('RGB').resize(image_size)
                img = np.array(img).flatten()
                inputs.append(img)
                targets.append(label)

        inputs = np.array(inputs)
        targets = np.array(targets)
        np.savez_compressed(saved_imgs_path, inputs=inputs, targets=targets)
        print(f'Successfully saved "{str(saved_imgs_path)}"')
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
    display_confusion_matrix(targets, predictions, plot_title=f'{data_partition} Confusion Matrix')

def display_confusion_matrix(target, predictions, labels=['Healthy', 'Unhealthy'], plot_title='Performance'):
    cm = confusion_matrix(target, predictions)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=labels)
    fig, ax = plt.subplots()
    cm_display.plot(ax=ax)
    ax.set_title(plot_title)
    plot_path = save_dir / f'{plot_title.lower().replace(" ", "_")}.png'
    plt.savefig(plot_path)
    plt.close()

def train_and_plot(argname, argvals, inputs_train, targets_train):
    plt.figure(figsize=(12, 8))

    models = []
    for argval in argvals:
        print(f"Training with {argname}={argval}")
        model = train_model(inputs_train, targets_train, **{argname: argval})
        plt.plot(model.loss_curve_, label=f'{argname}={argval}')
        models.append(model)

    plt.title(f'Loss Curve for Different {argname}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_dir / f'loss_curve_{argname}.png')

    return models

def plot_best_losses(models, argname, argvals):
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

def main(argname, argvals):
    inputs, targets = load_data()
    inputs_train, inputs_test, targets_train, targets_test = preprocess(inputs, targets)
    models = train_and_plot(argname, argvals, inputs_train, targets_train)
    for model in models:
        evaluate_model(model, 'Train', inputs_train, targets_train)
        evaluate_model(model, 'Test', inputs_test, targets_test)
    plot_best_losses(models, argname, argvals)

if __name__ == "__main__":
    main('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 50, 25)])
    main('learning_rate_init', [0.0001, 0.001, 0.01, 0.1])
