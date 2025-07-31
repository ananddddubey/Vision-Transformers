# Vision Transformer (ViT) for Hyperspectral Image Classification

This repository contains an implementation of a Vision Transformer (ViT) model for hyperspectral image (HSI) classification. The model processes hyperspectral datasets and classifies land cover types using self-attention mechanisms.

### Note:
This  is a sample code of a research project that is implemented from scratch without copying or cloning a GitHub repository. The Actual code is not showcased because the paper is under verification.

The actual code will be provided once the paper is published.

## Features
- Supports Indian Pines (IN), Salinas Valley (SV), and Pavia University (UP) datasets.
- Implements Vision Transformer (ViT) architecture.
- Preprocessing includes normalization, padding, and patch creation.
- Uses PyTorch for model training and evaluation.
- Utilizes GPU acceleration if available.

## Dataset
The following hyperspectral datasets are supported:
- **Indian Pines (IN)**
- **Salinas Valley (SV)**
- **Pavia University (UP)**

These datasets should be placed in the `./content/` directory before running the script.

## Installation
### Prerequisites
Ensure you have Python 3.7+ and install the required dependencies:
```bash
pip install numpy torch torchvision scikit-learn scipy matplotlib
```

## Usage
### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/HSI-ViT.git
   cd HSI-ViT
   ```
2. Ensure the datasets are placed in `./content/`.
3. Run the script:
   ```bash
   python main.py
   ```

### Configuration
Modify `dataset_name` in `main.py` to choose between datasets:
```python
dataset_name = 'IN'  # Options: 'IN', 'SV', 'UP'
```

## Model Architecture
The Vision Transformer (ViT) used in this project consists of:
- Patch embedding layer
- Transformer encoder with self-attention layers
- Classification head

## Training and Evaluation
The training process includes:
1. Data preprocessing (normalization and patch extraction)
2. Train-validation-test split
3. Training using Adam optimizer and cross-entropy loss
4. Evaluation on the test dataset

## Results
The trained model is evaluated using accuracy, loss, and other classification metrics. Modify the `train()` and `evaluate()` functions for custom evaluation.

## Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

