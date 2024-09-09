from sklearn.neural_network import MLPClassifier
from PIL import Image
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor, AdamW
import torch
import torch.nn as nn
import hashlib

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

    inputs = []
    targets = []

    if not saved_imgs_path.exists() or not return_array:
        for folder_name, label in class_folders.items():
            folder_path = base_path / folder_name
            filenames = sorted(p for p in folder_path.iterdir() if p.suffix == '.JPG')
            for filename in tqdm(filenames, desc=f'Processing {folder_name}'):
                inputs.append(filename)
                targets.append(label)

        if return_array:
            image_size = (128, 128)
            processed_inputs = []
            for img_path in inputs:
                img = Image.open(img_path).convert('RGB').resize(image_size)
                img = np.array(img).flatten()
                processed_inputs.append(img)
            inputs = np.array(processed_inputs)
            targets = np.array(targets)
            np.savez_compressed(saved_imgs_path, inputs=inputs, targets=targets)
            print(f'Successfully saved "{str(saved_imgs_path)}"')
        else:
            inputs = np.array(inputs)
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

class CLIPVisionClassifier(nn.Module):
    def __init__(self, vision_model, num_classes=2):
        super(CLIPVisionClassifier, self).__init__()
        self.vision_model = vision_model
        self.classifier = nn.Linear(self.vision_model.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # Take the pooled CLS token output
        logits = self.classifier(pooled_output)
        return logits

def fine_tune_clip_vision_model(img_paths, targets, num_epochs=5, batch_size=8):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    vision_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").vision_model
    model = CLIPVisionClassifier(vision_model).to("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = AdamW(model.parameters(), lr=5e-5)

    class PlantDataset(torch.utils.data.Dataset):
        def __init__(self, img_paths, targets, processor):
            self.img_paths = img_paths
            self.targets = targets
            self.processor = processor

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            img_path = self.img_paths[idx]
            target = self.targets[idx]
            img = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}  # Remove batch dimension
            return inputs, target

    dataset = PlantDataset(img_paths, targets, processor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = batch
            inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}
            labels = torch.tensor(labels).to("cuda" if torch.cuda.is_available() else "cpu")

            logits = model(**inputs)
            loss = torch.nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def classify_images_with_fine_tuned_model(model, img_paths, batch_size=30):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    predictions = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    for i in tqdm(range(0, len(img_paths), batch_size), desc="Classifying images"):
        batch_paths = img_paths[i:i + batch_size]
        images = [Image.open(str(path)).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            logits = model(**inputs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        predictions.extend(preds)

    return predictions

def calculate_accuracy(predictions, targets):
    correct = 0
    for prediction, target in zip(predictions, targets):
        if prediction == target:
            correct += 1
    accuracy = correct / len(targets)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

def plot_all(models, inputs_train, targets_train, inputs_test, targets_test, argname, argvals):
    plt.figure(figsize=(12, 8))

    for model, argval in zip(models, argvals):
        plt.plot(model.loss_curve_, label=f'{argname}={argval}')
    plt.title(f'Loss Curve for Different {argname}')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / f'loss_curve_{argname}.png')
    plt.close()

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

def main(argname=None, argvals=None):
    img_paths, targets = load_data(return_array=False)

    fine_tuned_model = fine_tune_clip_vision_model(img_paths, targets, num_epochs=3, batch_size=16)

    predictions = classify_images_with_fine_tuned_model(fine_tuned_model, img_paths, batch_size=32)

    calculate_accuracy(predictions, targets)

if __name__ == "__main__":
    main()
