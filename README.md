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
├── run.py              # Main training script
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

```
## Troubleshooting

### Debug Mode

Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Maintainer: [Wangshuran]
- Email: [1787054623@qq.com]

## Changelog

### v1.0.0 (2024-01-01)
- Initial release
- Basic gene regulatory prediction functionality
- DREAM4 dataset support
- Added cross-validation framework 
