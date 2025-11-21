# library imports
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

# local imports
from utils.main_utils import display_metrics, display_plot_roc, imshow, handle_error


# custom pytorch dataset
class CustomBinaryClassDataset(Dataset):
    def __init__(self, non_agri_dir, agri_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # load non-agri paths and assign label 0
        for fname in os.listdir(non_agri_dir):
            self.image_paths.append(os.path.join(non_agri_dir, fname))
            self.labels.append(0)
            
        # load agri paths and assign label 1
        for fname in os.listdir(agri_dir):
            self.image_paths.append(os.path.join(agri_dir, fname))
            self.labels.append(1)

        temp = list(zip(self.image_paths, self.labels))
        np.random.shuffle(temp)
        self.image_paths, self.labels = zip(*temp)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB') # ensure image is in RGB format
        label = self.labels[idx]
        
        # apply transformations if they exist
        if self.transform:
            image = self.transform(image)

        return image, label


def build_pytorch_model(
        conv_block_num=4,
        dense_block_num=2,
        filter_base=32,
        unit_base=128,
        kernel_size=5,
        padding='same',
        pool_size=2,
        dropout=0.4,
        n_channels=3,
        num_classes=2, 
        device='cpu', 
    ):
    try:
        print('Building and training PyTorch model...')
        if padding == 'same':                                   # calculate padding size for 'same' padding
            padding = kernel_size // 2                          # integer division for padding size
        layers = []                                             # to hold the layers of the model
        in_channels = n_channels                                # initial input channels (e.g., 3 for RGB)
        for conv_block in range(conv_block_num):                # create convolutional blocks
            out_channels = filter_base * (2 ** (conv_block))    # double the filters with each block (e.g., 32 -> 64 -> 128 -> ...)
            layers.extend([                                     # add convolutional blocks
                nn.Conv2d(                                      # convolutional layer
                    in_channels=in_channels,                    # input channels (e.g., 3 for RGB, then 32, 64, ...)
                    out_channels=out_channels,                  # output channels (e.g., 64, 128, ...)
                    kernel_size=kernel_size,                    # kernel size for convolution (e.g., 5)
                    padding=padding                             # padding for convolution (e.g., 'same' -> calculated padding, 0, 1, ...)
                ),
                nn.BatchNorm2d(num_features=out_channels),      # batch normalization layer -> number of features = out_channels (e.g., 32, 64, 128, ...)
                nn.ReLU(),                                      # activation layer (e.g., ReLU)
                nn.MaxPool2d(kernel_size=pool_size)             # max pooling layer
            ])
            in_channels = out_channels                          # update input channels for next block to current output channels (e.g., 32 -> 64 -> 128 -> ...)
        layers.extend([                                         # add transition block before dense layers
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),           # adaptive average pooling to reduce spatial dimensions ((1, 1) because we are flattening 2d to 1d next)
            nn.Flatten()                                        # flatten the output for dense layers 
            ])   
        in_features = out_channels                              # input features for dense layers = last out_channels (e.g., 128)
        for dense_block in range(dense_block_num):              # create dense blocks
            out_features = unit_base // (2 ** dense_block)      # halve the units with each block (e.g., 128 -> 64 -> 32 -> ...)
            layers.extend([                                     # add dense blocks
                nn.Linear(
                    in_features=in_features,                    # input features (e.g., 128, 256, ...)
                    out_features=out_features                   # output features (e.g., 256, 512, ...)
                ),                 
                nn.BatchNorm1d(num_features=out_features),      # batch normalization layer -> number of features = out_features (e.g., 128, 256, ...)
                nn.ReLU(),                                      # activation layer (e.g., ReLU)
                nn.Dropout(p=dropout)                           # dropout layer for regularization
            ])
            in_features = out_features                          # update input features for next block to current output features (e.g., 128 -> 256 -> ...)
        layers.append(                                          # final output layer
            nn.Linear(
                in_features=in_features, 
                out_features=num_classes
            )
        )
        model = nn.Sequential(*layers).to(device)               # create the model and move to device (CPU or GPU)
        return model
    except Exception as e:
        handle_error(e, def_name='build_pytorch_model')


def create_pytorch_custom_dataset(base_dir, dir_non_agri, dir_agri, batch_size=8):
    try:
        print('Creating PyTorch datasets...')
        imagefolder_dataset = datasets.ImageFolder(root=base_dir, transform=custom_pytorch_transform())
        print(f'Classes found by ImageFolder: {imagefolder_dataset.classes}')
        print(f'Class to index mapping: {imagefolder_dataset.class_to_idx}')

        # --- using your custom dataset ---
        custom_dataset = CustomBinaryClassDataset(dir_non_agri, dir_agri, transform=custom_pytorch_transform())
        custom_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # changed num_workers to 0 for compatibility

        return custom_loader
    except Exception as e:
        handle_error(e, def_name='create_pytorch_custom_dataset')


def create_pytorch_dataset(base_dir, batch_size=8):
    try:
        print('Creating PyTorch datasets...')
        imagefolder_dataset = datasets.ImageFolder(root=base_dir, transform=custom_pytorch_transform())
        print(f'Classes found by ImageFolder: {imagefolder_dataset.classes}')
        print(f'Class to index mapping: {imagefolder_dataset.class_to_idx}')

        # --- using the imagefolder dataset ---
        imagefolder_loader = DataLoader(imagefolder_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # changed num_workers to 0 for compatibility

        return imagefolder_loader
    except Exception as e:
        handle_error(e, def_name='create_pytorch_dataset')


def create_pytorch_loaders(dataset_path, worker, img_size=64, batch_size=128, train_split=.8, shuffle=True, num_workers=0):
    try:
        print('Creating PyTorch data loaders...')
        # transforms for training and validation
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, shear=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # create datasets and split into train/val
        full_dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        val_dataset.dataset.transform = val_transform

        # create data loaders and return
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=worker
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=worker
        )
        return train_loader, val_loader
    except Exception as e:
        handle_error(e, def_name='create_pytorch_loaders')


def custom_pytorch_transform():
    try:
        print('Creating custom transformations for PyTorch dataset...')
        custom_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(45),
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # normalize to [-1, 1]
        ])
        return custom_transform
    except Exception as e:
        handle_error(e, def_name='custom_transform')


def display_pytorch_batch(loader, batch_size, title):
    try:
        print(f'Displaying a PyTorch data batch for {title}...')

        # get one batch from the custom loader and display
        images, labels = next(iter(loader))
        print(f'Images batch shape: {images.shape}')
        print(f'Labels batch shape: {labels.shape}')
        plt.figure(figsize = (12, 6))
        for i in range(batch_size):
            plt.subplot(2, 4, i + 1)
            imshow(images[i])
            plt.title(f'{title}   Label:{labels[i].item()}')
            plt.axis('off')
    except Exception as e:
        handle_error(e, def_name='display_pytorch_batch')


def display_pytorch_history(acc_history, loss_history, model, val_loader, device):
    try:
        print('Displaying PyTorch training history and evaluating the model...')

        # plot accuracy history
        plt.figure(figsize=(12, 5))
        plt.plot(acc_history['train'], label='Train Acc')
        plt.plot(acc_history['val'], label='Val Acc')
        plt.title('Model Accuracy (PyTorch)')
        plt.legend()
        plt.show()

        # plot loss history
        plt.figure(figsize=(12, 5))
        plt.plot(loss_history['train'], label='Train Loss')
        plt.plot(loss_history['val'], label='Val Loss')
        plt.title('Model Loss (PyTorch)')
        plt.legend()
        plt.show()

        # evaluate model on validation set
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f'The accuracy of the model is: {accuracy:.4f}')
    except Exception as e:
        handle_error(e, def_name='display_pytorch_history_and_evaluation')

    
def evaluate_pytorch_model(model, val_loader, device, model_name):
    try:
        print('Evaluating PyTorch model to display metrics...')
        y_true = []
        y_pred = []
        y_prob = []
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                # Get probabilities using softmax for CrossEntropyLoss
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get predicted labels
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_prob.extend(probabilities.cpu().numpy())

        # Convert lists to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        # Get class labels from the dataset
        class_labels = val_loader.dataset.dataset.classes

        # display metrics and ROC curve
        display_metrics(y_true, y_pred, y_prob[:, 1], class_labels, model_name)
        y_prob_positive = y_prob[:, 1]              # probabilities for the positive class
        display_plot_roc(y_true, y_prob_positive, model_name)
    except Exception as e:
        handle_error(e, def_name='evaluate_pytorch_model')


def pytorch_iteration(loader, model, optimizer, criterion, device, epoch, epochs, criterion_method, loader_type):
    if loader_type == 'train':
        model.train()
    else: # 'val'
        model.eval()
    loss, correct, samples = 0, 0, 0
    for _, (images, labels) in enumerate(tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}')):
        images = images.to(device)                              # move images to device
        accurate_labels = labels.to(device)                     # move labels to device  
        if criterion_method == 'bce':
            loss_labels = accurate_labels.float().unsqueeze(1)  # convert labels to float and reshape for BCEWithLogitsLoss
        else: # 'cross_entropy'
            loss_labels = accurate_labels                       # keep labels as integer class indices for crossentropyloss   
        if loader_type == 'train': 
            optimizer.zero_grad()
        outputs = model(images)                                 # outputs are raw logits

        loss_tensor = criterion(outputs, loss_labels)           # compute loss tensor
        if loader_type == 'train':
            loss_tensor.backward()                              # backpropagate loss
            optimizer.step()
        loss += loss_tensor.item()                              # get scalar loss value
        if criterion_method == 'bce':                           # if using BCEWithLogitsLoss for binary classification
            preds = (outputs > 0).long().squeeze(1)             # convert logits to binary predictions
        else: # 'cross_entropy'
            preds = torch.argmax(outputs, dim=1)                # get predicted class indices            
        correct += (preds == accurate_labels).sum().item()
        samples += accurate_labels.size(0)
    return loss, correct, samples


def pytorch_training_loop(loss_function, lr, epochs, model, train_loader, val_loader, device, model_name):
    try:
        print('Starting PyTorch training loop...')
        criterion = loss_function                               # loss function for binary classification
        optimizer = optim.Adam(model.parameters(), lr=lr)       # adam optimizer
        best_loss = float('inf')                                # initialize best loss to infinity
        loss_history = {'train': [], 'val': []}                 # to store loss history
        acc_history = {'train': [], 'val': []}                  # to store accuracy history
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            criterion_method = 'bce'
        else: # nn.CrossEntropyLoss
            criterion_method = 'cross_entropy'
        for epoch in range(epochs):
            start_time = time.time()                                    # to get the training time for each epoch

            # training phase
            train_loss, train_correct, train_total = pytorch_iteration(
                train_loader,                                        
                model,                                                       
                optimizer,                                            
                criterion, 
                device, 
                epoch, 
                epochs, 
                criterion_method, 
                loader_type='train', 
            )
            if device == 'cuda': torch.cuda.synchronize()               # synchronize cuda before stopping timer (if using gpu)

            # validation phase
            with torch.no_grad():                                       # disable gradient calculation for validation
                val_loss, val_correct, val_total = pytorch_iteration(
                    val_loader, 
                    model, 
                    optimizer, 
                    criterion, 
                    device, 
                    epoch, 
                    epochs, 
                    criterion_method, 
                    loader_type='val', 
                )
        
            # save the best model
            avg_val_loss = val_loss / len(val_loader)
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), model_name)
            
            # store metrics
            loss_history['train'].append(train_loss / len(train_loader))
            loss_history['val'].append(val_loss / len(val_loader))
            acc_history['train'].append(train_correct / train_total)
            acc_history['val'].append(val_correct / val_total)
            
            # print epoch summary
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Train Loss: {loss_history["train"][-1]:.4f} | Val Loss: {loss_history["val"][-1]:.4f}')
            print(f'Train Acc: {acc_history["train"][-1]:.4f} | Val Acc: {acc_history["val"][-1]:.4f}')
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1} training completed in {epoch_time:.2f} seconds\n')
        return model, loss_history, acc_history
    except Exception as e:
        handle_error(e, def_name='pytorch_training_loop')


def set_pytorch_processing_env():
    try:
        print('Setting PyTorch processing environment...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Device used is {device}')
        return device
    except Exception as e:
        handle_error(e, def_name='set_pytorch_processing_env')


def set_pytorch_seed(seed_value):
    try:
        print(f'Setting PyTorch seed to {seed_value} for reproducibility...')

        # python and numpy
        random.seed(seed_value)
        np.random.seed(seed_value)

        # PyTorch (CPU & GPU)
        torch.manual_seed(seed_value)            
        torch.cuda.manual_seed_all(seed_value)   

        # cuDNN: force repeatable convolutions
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        handle_error(e, def_name='set_pytorch_seed')

def visualize_satellite_agriculture(
        model,
        dir_non_agri,
        dir_agri,
        device,
        grid_size=4,
        tile_size=64,
        save_path=None
    ):
    try:
        print('Generating agricultural tile grid visualization...')

        # ensure save_path is provided
        if save_path is None:
            raise ValueError('save_path must not be None. Provide a valid output file path.')

        # ensure save_path is an image file
        valid_ext = ('.png', '.jpg', '.jpeg')
        if not save_path.lower().endswith(valid_ext):
            print(f'[WARNING] The provided save_path "{save_path}" is not an image file.')
            print('          Automatically converting output to PNG format...')
            save_path = save_path + '.png'

        # collect image paths
        non_agri_files = []
        for f in os.listdir(dir_non_agri):
            non_agri_files.append(os.path.join(dir_non_agri, f))
        agri_files = []
        for f in os.listdir(dir_agri):
            agri_files.append(os.path.join(dir_agri, f))

        # combine and shuffle
        all_files = non_agri_files + agri_files
        random.shuffle(all_files)

        # pick N tiles for the grid
        num_tiles = grid_size * grid_size
        chosen_files = all_files[:num_tiles]

        # preprocessing pipeline
        preprocess = transforms.Compose([
            transforms.Resize((tile_size, tile_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])

        model.eval()

        # create output grid canvas
        canvas_size = grid_size * tile_size
        grid_img = Image.new('RGB', (canvas_size, canvas_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(grid_img)

        # loop over tiles and classify each
        index = 0
        for row in range(grid_size):
            for col in range(grid_size):
                img_path = chosen_files[index]
                index += 1
                tile = Image.open(img_path).convert('RGB')
                tile_tensor = preprocess(tile).unsqueeze(0).to(device)

                # predict
                with torch.no_grad():
                    output = model(tile_tensor)
                    prob = torch.softmax(output, dim=1)[0, 1].item()
                    label = 'AGRI' if prob >= 0.5 else 'NON-AGRI'

                # paste tile into grid
                x = col * tile_size
                y = row * tile_size
                grid_img.paste(tile, (x, y))

                # draw label background (green for agri)
                if prob >= 0.5:
                    color = (0, 200, 0)   # green
                else:
                    color = (200, 0, 0)   # red

                draw.rectangle([x, y, x + tile_size, y + 20], fill=color)
                draw.text((x + 4, y + 2), f'{label} {prob:.2f}', fill=(255, 255, 255))
        # save final grid
        grid_img.save(save_path)
        print(f'Tile grid saved to: {save_path}')
        return grid_img
    except Exception as e:
        handle_error(e, def_name='visualize_satellite_agriculture')


def worker_init_fn(worker_id, seed):
    try:
        print(f'Initializing worker {worker_id} with seed {seed}...')
        worker_seed = seed + worker_id
        np.random.seed(worker_seed) 
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    except Exception as e:
        handle_error(e, def_name='worker_init_fn')