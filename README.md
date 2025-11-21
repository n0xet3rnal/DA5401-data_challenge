# DA5401 Data Challenge - Two-Stage Neural Network for Score Prediction

## Project Overview
This project implements a **two-stage deep learning pipeline** for predicting quality scores of LLM responses based on evaluation metrics. The pipeline combines **metric learning** with **score calibration** to achieve accurate predictions.

### Architecture
1. **Stage 1**: Metric Learning using ranked triplet loss (2-Tower Specialist Neural Network)
2. **Stage 2**: Score Calibration using regression (Simple Regressor Neural Network)

## File Structure
```
├── training_pipeline.ipynb      # Main two-stage training pipeline
├── EDA.ipynb                    # Exploratory data analysis
├── PCA.ipynb                    # PCA analysis and visualization
├── project_utils.py             # Data preprocessing utilities
├── nn_models.py                 # Neural network model architectures
├── datasets.py                  # PyTorch dataset implementations
├── utils.py                     # Training utilities and helpers
├── advanced_losses.py           # Custom loss functions
├── README.md                    # This file
├── environment.yml              # Conda environment specification
├── data/
│   ├── Given the train_data file, embedder can generate vectors
├── experiments/
│   ├── EDA.ipynb                # Data exploration experiments
│   ├── compression.ipynb        # Dimensionality reduction experiments
│   └── model_testing.ipynb      # Model evaluation experiments
└── images/
    
```

## Pipeline Components

### Stage 1: Metric Learning (Triplet Loss)
**Objective**: Learn a latent embedding space where similar metric-response pairs are close together

- **Model**: FinalSpecialistNN (2-Tower Architecture)
  - Tower 1: Encodes evaluation metrics (rules)
  - Tower 2: Encodes LLM responses (events)
  - Latent dimension: 128
- **Loss Function**: Ranked Triplet Loss with dynamic margins
- **Training Strategy**: 
  - Batch size: 256
  - Epochs: 100
  - Learning rate: 0.00076
  - Optimizer: Adam
- **Output**: Frozen embedding model saved as `frozen_specialist_model.pth`

### Stage 2: Score Calibration (Regression)
**Objective**: Map learned embeddings to actual quality scores

- **Model**: SimpleRegressorNN (MLP)
  - Input: Concatenated features [rule_embed, event_embed, gap_embed] (384 dims)
  - Output: Predicted score (continuous)
- **Loss Function**: Mean Squared Error (MSE)
- **Training Strategy**:
  - Batch size: 128
  - Epochs: 300
  - Learning rate: 0.00037
  - Optimizer: Adam
- **Validation Metrics**: MSE, RMSE, Accuracy (rounded predictions)
- **Output**: Final regressor model saved as `final_regressor_model.pth`

## Data Processing

### Input Data
The dataset contains LLM response evaluations with the following fields:
- **metric_name**: Evaluation metric (e.g., "coherence/clarity", "accuracy/factual")
- **user_prompt**: User's input prompt
- **system_prompt**: System instruction for the LLM
- **response**: LLM's generated response
- **score**: Quality score (1-10 scale) - *training only*

### Embedding Generation
Text fields are embedded using **SentenceTransformer** (`google/embeddinggemma-300m`):
- Metric names → 768-dimensional vectors
- User prompts → 768-dimensional vectors
- System prompts → 768-dimensional vectors
- Responses → 768-dimensional vectors

### Dimensionality Reduction
**PCA** is applied to reduce computational complexity:
- **Metric embeddings**: 768 → 87 components
- **Combined prompt/response**: 2304 → 827 components
- Transformers saved in `pca_transformers.pkl` for inference

### Data Splits
- **Training set**: 80% of data (stratified by metric_id)
- **Validation set**: 20% of data (stratified by metric_id)
- **Singleton removal**: Metrics with only one sample are dropped

## Training Visualizations

The pipeline generates **loss plots** for both stages:
- **Stage 1**: Train vs Validation Triplet Loss over 100 epochs
- **Stage 2**: Train vs Validation MSE over 300 epochs
- Plots use **seaborn** styling with whitegrid theme
- Final train and validation losses are reported for both stages

## Inference Pipeline

The `predict_from_embedded_with_position_ids()` function handles test predictions:
1. Load pre-embedded test data
2. Apply saved PCA transformations
3. Load frozen Stage 1 and Stage 2 models
4. Generate embeddings using Stage 1 model
5. Predict scores using Stage 2 regressor
6. Round predictions to integers
7. Save results to `output.csv` with format: `ID, score`

## Model Architecture Details

### FinalSpecialistNN (Stage 1)
```python
Tower1 (Rule Encoder):
  - Input: PCA-reduced metric embeddings (87 dims)
  - Hidden layers with batch normalization and dropout
  - Output: 128-dimensional latent embedding

Tower2 (Event Encoder):
  - Input: PCA-reduced prompt/response embeddings (827 dims)
  - Hidden layers with batch normalization and dropout
  - Output: 128-dimensional latent embedding
```

### SimpleRegressorNN (Stage 2)
```python
Input: [rule_embed || event_embed || gap_embed] (384 dims)
Hidden layers: MLP with ReLU activations
Output: Single continuous score value
```

## How to Use

### 1. Environment Setup
```bash
conda env create -f environment.yml
conda activate da5401-challenge
```

### 2. Data Preparation (if needed)
```python
from project_utils import Embedder

# Embed training data
embedder_train = Embedder(
    data_file='data/train_data.json',
    output_file='data/train_data_embedded.json',
    is_test=False
)
embedder_train.process()

# Embed test data
embedder_test = Embedder(
    data_file='data/test_data.json',
    output_file='data/test_data_embedded.json',
    is_test=True
)
embedder_test.process()
```

### 3. Training
Open and run `training_pipeline.ipynb`:
- Loads embedded training data
- Applies PCA transformations
- Trains Stage 1 (metric learning)
- Trains Stage 2 (score calibration)
- Generates loss plots for both stages
- Saves both models

### 4. Inference
The notebook includes inference at the end:
```python
predict_from_embedded_with_position_ids(
    './data/test_data_embedded.json', 
    'output.csv'
)
```



## Requirements

### Core Dependencies
- Python 3.10+
- PyTorch 2.0+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- sentence-transformers
- tqdm, joblib

## Performance Monitoring

### Stage 1 Metrics
- Train Loss (Triplet)
- Validation Loss (Triplet)
- Average margin values

### Stage 2 Metrics
- Train MSE
- Validation MSE
- Validation RMSE
- Validation Accuracy (rounded predictions)

## Model Outputs

1. **frozen_specialist_model.pth**: Stage 1 embedding model
2. **final_regressor_model.pth**: Stage 2 score predictor
3. **pca_transformers.pkl**: Fitted PCA transformers
4. **output.csv**: Test predictions (ID, score)

## Author

*Jerry Jose*  
*DA5401 Data Challenge*  
*IIT Madras*