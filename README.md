# Agriculture Detection with Deep Learning

A Python project that uses deep learning models (Keras and PyTorch) to classify satellite images for agricultural and non-agricultural land use.

## Author

**Agua Chile**
- GitHub: [@agua_chile](https://github.com/aguachile)
- Project: agriculture_detection

## Acknowledgements

This project is based on the IBM Coursera course:
**[AI Capstone Project with Deep Learning](https://www.coursera.org/learn/ai-deep-learning-capstone)**

Special thanks to IBM for providing the dataset and foundational knowledge for this capstone project.

## Features

- 🛰️ **Satellite Image Classification**: Classifies images as agricultural or non-agricultural land.
- 🧠 **Dual Framework Implementation**: Provides complete workflows for both TensorFlow/Keras and PyTorch.
- 🚀 **Performance Optimized**: Includes data augmentation, caching, and prefetching for efficient training.
- 📊 **Rich Evaluation**: Generates detailed performance metrics, including accuracy, ROC-AUC, confusion matrices, and classification reports.
- 📈 **Visualization**: Plots training history and ROC curves to visualize model performance.
- ⚙️ **Customizable Architecture**: Easily configurable neural network architecture (number of layers, filter sizes, etc.).

## Technology Stack

- **Deep Learning Frameworks**: TensorFlow/Keras, PyTorch
- **Data Handling**: NumPy, Pillow
- **Visualization**: Matplotlib
- **Metrics**: Scikit-learn
- **Environment**: Python, Jupyter Notebook

## Setup

### Prerequisites

- Python 3.12
- Jupyter Notebook or VS Code with Jupyter extension

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/agua-chile/agriculture_detection
   cd agriculture_detection
   ```

2. **Create and activate virtual environment and install dependencies**
   ```bash
   cd env
   chmod +x setup.sh && ./setup.sh
   ```

3. **Run ipynb files**
   ```bash
   Go into each .ipynb file, select the .venv in env/ and run the cells to execute the Keras and PyTorch workflows.
   ```

## Usage

1. **Launch Jupyter Notebook or open in VS Code**
   - Navigate to the `agriculture_detection/` directory.

2. **Run the Keras notebook**
   - Open and run the cells in `keras_ai.ipynb` to train and evaluate the TensorFlow/Keras model.

3. **Run the PyTorch notebook**
   - Open and run the cells in `pytorch_ai.ipynb` to train and evaluate the PyTorch model.

The notebooks will download the dataset, preprocess the images, build the models, train them, and display the evaluation results.

## Project Structure

```
agriculture_detection/
├── README.md                # Project documentation
├── .gitignore               # Git ignore rules
├── keras_ai.ipynb           # Keras training and evaluation workflow
├── pytorch_ai.ipynb         # PyTorch training and evaluation workflow
├── env/                     # Environment bootstrap scripts and pinned deps
│   ├── .env                 # Environment variables (not tracked)
│   ├── .venv/               # Virtual environment (not tracked)
│   ├── requirements.txt     # Package requirements for the project venv
│   └── setup.sh             # Helper script to create and seed the virtual env
└── utils/                   # Shared Python utilities used by both notebooks
    ├── keras_ai_utils.py    # Helper functions for the Keras pipeline
    ├── main_utils.py        # Common helpers for metrics, plotting, etc.
    └── pytorch_ai_utils.py  # Helper functions for the PyTorch pipeline
```

## Configuration

Hyperparameters for both models can be configured within their respective notebooks (`keras_ai.ipynb` and `pytorch_ai.ipynb`). Key parameters include:

- **Image Dimensions**: `img_w`, `img_h`
- **Batch Size**: `batch_size`
- **Learning Rate**: `lr`
- **Epochs**: `n_epochs`
- **Network Architecture**: `conv_block_num`, `dense_block_num`, `filter_base`, `unit_base`
- **Data Augmentation**: `rotation_range`, `zoom_range`, etc.

## Features in Detail

### Data Processing and Augmentation
- The dataset is automatically downloaded and extracted.
- `ImageDataGenerator` (Keras) and `transforms` (PyTorch) are used to apply on-the-fly data augmentation, including rotations, flips, and zooms to improve model generalization.

### Model Architecture
- Both models use a similar CNN architecture consisting of multiple convolutional blocks followed by dense layers.
- Each convolutional block contains a `Conv2D` layer, `BatchNormalization`, and `MaxPooling2D`.
- The architecture is highly customizable through hyperparameters defined in the notebooks.

### Training and Evaluation
- The models are trained using their respective framework's training loops.
- `ModelCheckpoint` is used in Keras to save the best model based on validation accuracy.
- After training, the models are evaluated on a validation set, and key metrics are displayed, including a classification report and confusion matrix.

## License

This project is licensed under the MIT License.

## Support

For questions or issues:
- Review the error messages and tracebacks printed in the notebook.
- Ensure all dependencies from `requirements.txt` are correctly installed.
- Verify that the dataset was downloaded and extracted correctly into the `data/` directory.

---

*Built using Keras, TensorFlow, and PyTorch*
