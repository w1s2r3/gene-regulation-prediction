# Deep Learning Model for Gene Regulatory Network Prediction

## Project Overview

This project implements a deep learning model for gene regulatory network prediction, specifically designed for analyzing E. coli gene expression data. The model integrates temporal, static, and topological features, and uses multi-scale attention mechanisms and graph convolutional networks to predict regulatory relationships between genes.

## Key Features

- **Multi-modal Feature Fusion**: Integrates temporal expression data, static features, and network topology information
- **Advanced Network Architecture**: Utilizes Graph Convolutional Attention (GCA) layers and multi-scale temporal feature extraction
- **Adaptive Loss Function**: Optimized for gene regulatory prediction tasks
- **Cross-validation Support**: Complete k-fold cross-validation framework

## Project Structure

```
├── run_ecoli.py              # Main training script
├── model.py                  # Deep learning model definition
├── data_loader.py            # Data loading and preprocessing
├── load_data.py              # Dataset loading utility
├── cross_validate_ecoli.py   # Cross-validation script
├── data_config.json          # Dataset configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Installation Guide

### Requirements

- Python 3.8+
- CUDA 11.0+ (recommended for GPU acceleration)
- 8GB+ RAM

### Installation Steps

1. Clone the repository:
```bash
git clone <your-repository-url>
cd gene-regulation-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install PyTorch Geometric (if needed):
```bash
pip install torch-geometric
```

## Data Preparation

### Data Format

The project supports DREAM4-format datasets, including:
- Time series data (`.tsv` format)
- Gold standard data (gene regulatory relationships)
- Wildtype data
- Multifactorial data

### Data Configuration

Configure dataset paths in `data_config.json`:

```json
{
    "DATASET_PATHS": {
        "Ecoli": {
            "base_dir": "data/Ecoli1",
            "gold_standard_dir": "data/Ecoli1",
            "file_prefix": "",
            "gold_standard_prefix": ""
        }
    }
}
```

## Usage

### Basic Training

Run the main training script:

```bash
python run_ecoli.py
```

### Cross-validation

Perform k-fold cross-validation:

```bash
python cross_validate_ecoli.py
```

## Model Architecture

### Core Components

1. **Multi-scale Temporal Feature Extraction**: Processes time series data with different scales
2. **Graph Convolutional Attention Layer (GCA)**: Combines graph structure and attention mechanism
3. **Feature Fusion Module**: Fuses temporal, static, and topological features
4. **Adaptive Loss Function**: Optimized for imbalanced data

### Network Structure

```
Input Data
    ↓
Temporal Feature Extraction → Static Feature Extraction → Topological Feature Extraction
    ↓                        ↓                          ↓
Feature Fusion Layer
    ↓
Graph Convolutional Attention Layer
    ↓
Fully Connected Layer
    ↓
Output Prediction
```

## Evaluation Metrics

The model is evaluated using the following metrics:
- **AUC-ROC**: Area under the Receiver Operating Characteristic curve
- **Average Precision**
- **F1-Score**
- **Precision-Recall Curve**

## Experimental Results

Typical performance on the E. coli dataset:
- AUC-ROC: 0.85+
- Average Precision: 0.80+
- F1-Score: 0.75+

## Configuration Parameters

Main configuration parameters in `run_ecoli.py`:

```python
# Model parameters
hidden_channels = 256
num_layers = 4
num_heads = 8
dropout = 0.2

# Training parameters
batch_size = 32
learning_rate = 0.001
epochs = 100
early_stop_patience = 15
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch_size or use CPU for training
2. **Data loading errors**: Check data paths and formats
3. **Dependency conflicts**: Use a virtual environment

### Debug Mode

Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contribution Guide

Contributions are welcome! Please submit Issues and Pull Requests.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@article{gene_regulation_prediction,
  title={Deep Learning for Gene Regulation Network Prediction},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
```

## Contact

- Maintainer: [Your Name]
- Email: [your.email@example.com]
- Project homepage: [GitHub Repository URL]

## Changelog

### v1.0.0 (2024-01-01)
- Initial release
- Basic gene regulatory prediction functionality
- E. coli dataset support
- Added cross-validation framework 