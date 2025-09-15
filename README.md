# GPU Workload Classification with XGBoost and Transformer Models

Welcome to the GPU Workload Classification project! This repository contains a comprehensive solution to classify GPU workloads into three categories: **Gaming**, **AI Training**, and **Rendering** based on various GPU metrics. Leveraging both traditional machine learning with XGBoost and state-of-the-art deep learning using Transformer networks, this project showcases high-performance classification models on synthetic GPU workload data.

***

## Project Overview

GPUs run diverse workloads that exhibit distinct utilization patterns, memory usage, temperature, power consumption, and clock speeds. Correctly identifying these workloads can help optimize performance, manage thermal profiles, and predict power usage efficiently.

This project includes:

- A **synthetic dataset** generator simulating GPU metrics for Gaming, AI Training, and Rendering workloads.
- Preprocessing, normalization, and train-test split pipelines.
- Implementation and training of:
  - An **XGBoost classifier** for tabular data classification.
  - A **Transformer-based deep learning model** tailored for sequence data classification.
- Model evaluation with precision, recall, and F1-scores.
- Saving model and scaler artifacts for easy reproducibility.

***

## Dataset

The synthetic dataset contains the following features for each workload sample:

| Feature      | Description                 |
|--------------|-----------------------------|
| gpu_util     | GPU utilization percentage  |
| vram_usage   | VRAM usage in MB            |
| temp         | GPU temperature in °C       |
| power        | Power consumption in Watts  |
| core_clock   | GPU core clock speed in MHz |
| task_type   | Workload category (label)    |

The dataset simulates realistic value ranges for each workload category and contains 900 samples (300 per workload).

***

## Installation

To get started, clone the repo and install the required dependencies:

```bash
pip install -r requirements.txt
```

*Requirements include: pandas, numpy, scikit-learn, xgboost, torch, torchvision.*

***

## Usage

### Data Generation & Preprocessing

- Synthetic data is generated programmatically by simulating each workload's GPU characteristics.
- Features are normalized using `StandardScaler`.
- Labels are encoded to integer classes with `LabelEncoder`.

### Training XGBoost Classifier

Train a gradient boosted tree model optimized for tabular data classification:

```python
import xgboost as xgb

clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
clf.fit(X_train, y_train)
```

### Training Transformer Classifier

Train a deep Transformer model on sequence data (time series of metrics):

```python
class TransformerClassifier(nn.Module):
    ...
model = TransformerClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    ...
```

### Evaluation

Both models output classification reports evaluating:

- Precision
- Recall
- F1-Score
- Accuracy

Transformer model achieves near-perfect classification, with 100% accuracy on test data.

***

## Results Summary

| Workload    | XGBoost F1-Score | Transformer F1-Score |
|-------------|------------------|---------------------|
| AI Training | 0.94             | 1.00                |
| Gaming      | 0.95             | 1.00                |
| Rendering   | 0.88             | 1.00                |
| **Overall Accuracy** | **92%**   | **100%**             |

***

## File Structure

- `synthetic_gpu_workload.csv` - The generated synthetic dataset.
- `gpu_task_classifier.pth` - Saved transformer model weights.
- `scaler.save` - Saved scaler for input feature normalization.
- `train_xgboost.py` - Script for training and evaluating XGBoost classifier.
- `train_transformer.py` - Script for training the Transformer model.
- `requirements.txt` - Python dependencies.

***

## Contributing

Contributions and suggestions to improve dataset realism, model architecture, or add more workloads are welcome. Please create issues or pull requests.

***

## License

This project is licensed under the MIT License.

***

## Contact

For questions or collaboration inquiries, please reach out to the repository maintainer.

***

Thank you for exploring this GPU workload classification project. Harness the power of machine learning to intelligently understand GPU usage patterns!

[1](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/83134720/655d9dcf-dfc8-4dd8-8ff5-cd0e49939234/vertopal.com_GPU-1.pdf)
