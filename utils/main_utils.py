# import libraries
import os
import random
import tarfile
import httpx
from pathlib import Path
import skillsnetwork
import numpy as np
import matplotlib.pyplot as plt
import traceback
import sys
from textwrap import indent
from colorama import Fore, Style, init
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
init(autoreset=True)      # Initialize Colorama (auto-reset ensures colors don't 'leak')


async def check_skillnetwork_extraction(extract_dir, url):
    try:
        print('Checking write permissions for skillsnetwork extraction...')
        symlink_test = os.path.join(extract_dir, 'symlink_test')
        if not os.path.exists(symlink_test):
            os.symlink(os.path.join(os.sep, 'tmp'), symlink_test) 
            print('Write permissions available for downloading and extracting the dataset tar file')
        os.unlink(symlink_test)
        await skillsnetwork.prepare(url=url, path=extract_dir, overwrite=True)
    except Exception as err:
        handle_error(err, def_name='check_skillnetwork_extraction', msg='Primary download/extraction method failed., falling back to manual download and extraction.')
        file_name = Path(url).name
        tar_path = os.path.join(extract_dir, file_name)
        print(f'tar_path: {os.path.exists(tar_path)} ___ {tar_path}')
        await download_tar_dataset(url, tar_path, extract_dir, file_name)


def display_metrics(y_true, y_pred, y_prob, class_labels, model_name):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'Loss': log_loss(y_true, y_prob),
        'F1 Score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'Confusion Matrix': confusion_matrix(y_true, y_pred),
        'Classification Report': classification_report(y_true, y_pred, target_names=class_labels, digits=4),
        'Class labels': class_labels
    }
    print(f'Evaluation metrics for the \033[1m{model_name}\033[0m')
    print(f'Accuracy: {"":<1}{metrics["Accuracy"]:.4f}')
    print(f'ROC-AUC: {"":<2}{metrics["ROC-AUC"]:.4f}')
    print(f'Loss: {"":<5}{metrics["Loss"]:.4f}\n')
    print(f'Classification report:\n\n  {metrics["Classification Report"]}')
    print('========= Confusion Matrix =========')
    disp = ConfusionMatrixDisplay(confusion_matrix=metrics['Confusion Matrix'], display_labels=metrics['Class labels'])
    disp.plot()
    plt.show()


def display_plot_roc(y_true, y_prob, model_name):
    n_classes = y_prob.shape[1] if y_prob.ndim > 1 else 1
    if n_classes == 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
    else:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f'{model_name} class {i} (AUC = {auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()


async def download_tar_dataset(url, tar_path, extract_dir, file_name):
    try:
        print('Downloading tar dataset...')
        if not os.path.exists(tar_path): # download only if file not downloaded already
            try:
                print(f'Downloading from {url}...')
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, follow_redirects=True)# Download the file asynchronously
                    response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                
                    with open(tar_path , 'wb') as f:
                        f.write(response.content) # Save the downloaded file
                    print(f'Successfully downloaded "{file_name}".')
            except httpx.HTTPStatusError as http_err:
                print(f'HTTP error occurred during download: {http_err}')
            except Exception as download_err:
                print(f'An error occurred during the fallback process: {download_err}')
        else:
            print(f'dataset tar file already downloaded at: {tar_path}')
        with tarfile.open(tar_path, 'r:*') as tar_ref:
            tar_ref.extractall(path=extract_dir)
        print(f'Successfully extracted to "{extract_dir}".')
    except Exception as e:
        handle_error(e, def_name='download_tar_dataset', msg='An error occurred during extraction of the dataset tar file.')


def get_file_names(dataset_path):
    try:
        print('Getting file names from dataset...')
        fnames = []
        for dirname, _, filenames in os.walk(dataset_path):
            for filename in filenames:
                fnames.append(os.path.join(dirname, filename))
        print(f'total files in dataset: {len(fnames)}')
        nfname_print=2
        for f in fnames[:nfname_print]:
            print(f)
        for f in fnames[-nfname_print:]:
            print(f)
        return fnames
    except Exception as e:
        handle_error(e, def_name='get_file_names')


def get_random_sample_image(dir_non_agri, dir_agri):
    try:
        print('Selecting a random image from dataset...')

        # collect all image file paths
        non_agri_files = []
        for f in os.listdir(dir_non_agri):
            non_agri_files.append(os.path.join(dir_non_agri, f))
        agri_files = []
        for f in os.listdir(dir_agri):
            agri_files.append(os.path.join(dir_agri, f))

        all_files = non_agri_files + agri_files

        if len(all_files) == 0:
            raise FileNotFoundError("No images found in the provided directories.")

        # choose one random image path
        chosen_image = random.choice(all_files)

        print(f'Random sample selected: {chosen_image}')
        return chosen_image

    except Exception as e:
        handle_error(e, def_name='get_random_sample_image')




def handle_error(exc, def_name, msg=''):
    exc_type, exc_value, exc_tb = sys.exc_info()
    tb_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print('\n' + Fore.RED + '=' * 100)
    print(Fore.RED + Style.BRIGHT + f'🚨  ERROR OCCURRED IN {def_name}()')
    if msg:
        print(Fore.YELLOW + f'Context: {msg}')
    print(Fore.RED + '-' * 100)
    print(Fore.CYAN + f'Type: {exc_type.__name__}')
    print(Fore.MAGENTA + f'Message: {exc}')
    print(Fore.RED + '-' * 100)
    print(Fore.WHITE + Style.BRIGHT + 'Traceback (most recent call last):\n')
    print(indent(Fore.LIGHTBLACK_EX + tb_str.strip(), '  '))
    print(Fore.RED + '=' * 100 + Style.RESET_ALL + '\n')
    print('Full Raw Traceback:')
    print(tb_str)


def imshow(img):
    try:
        img = img / 2 + 0.5  # Un-normalize from [-1, 1] to [0, 1]
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0))) # Convert from C,H,W to H,W,C
    except Exception as e:
        handle_error(e, def_name='imshow')


def shuffle_data(dir_non_agri, dir_agri):
    try:
        print('Shuffling data...')
        all_image_paths = []
        all_labels = []
        label_non_agri = 0
        label_agri = 1
        for fname in os.listdir(dir_non_agri):
            all_image_paths.append(os.path.join(dir_non_agri, fname))
            all_labels.append(label_non_agri)
        for fname in os.listdir(dir_agri):
            all_image_paths.append(os.path.join(dir_agri, fname))
            all_labels.append(label_agri)
        temp = list(zip(all_image_paths, all_labels))
        np.random.shuffle(temp)
        all_image_paths, all_labels = zip(*temp)
        return all_image_paths, all_labels
    except Exception as e:
        handle_error(e, def_name='shuffle_data')