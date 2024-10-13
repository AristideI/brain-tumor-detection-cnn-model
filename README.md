# brain-tumor-detection-cnn-model

This notebook uses deep learning to classify MRI and X-ray images for early brain tumor and lung cancer detection. Developed with TensorFlow and Keras, it addresses diagnostic challenges in low-resource healthcare settings like Rwanda.

# Training Report

## Model Training Report

## 1. Data Preprocessing

- **Dataset Location**:
  - Training: `data/training`
  - Validation: `data/validation`
- **Image Preprocessing**:

  - **Grayscale Conversion**: All images are converted to grayscale.
  - **Rescaling**: All images were rescaled by a factor of `1/255`.
  - **Data Augmentation**: Applied random transformations to training images, including rotation, width/height shifts, shearing, zooming, and horizontal flipping.
  - **Validation Split**: 20% of the training dataset was reserved for validation.

- **Batch Size**: 32
- **Image Dimensions**: 64x64 pixels

## 2. Model Architectures

### Model 1: Vanilla Model (No Optimization Techniques)

The first model is a simple Convolutional Neural Network (CNN) without additional regularization or optimization techniques. The architecture is as follows:

| Layer Type   | Filters | Kernel Size | Activation | Pool Size | Dropout Rate |
| ------------ | ------- | ----------- | ---------- | --------- | ------------ |
| Conv2D       | 32      | (3, 3)      | ReLU       | -         | -            |
| MaxPooling2D | -       | -           | -          | (2, 2)    | -            |
| Conv2D       | 64      | (3, 3)      | ReLU       | -         | -            |
| MaxPooling2D | -       | -           | -          | (2, 2)    | -            |
| Flatten      | -       | -           | -          | -         | -            |
| Dense        | 64      | -           | ReLU       | -         | -            |
| Dense        | 32      | -           | ReLU       | -         | -            |
| Output       | -       | -           | Softmax    | -         | -            |

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall

**Training Results**:

- **Final Training Accuracy**: 100.0%
- **Final Validation Accuracy**: 91.56%

### Model 2: Optimized Model with Regularization and Dropout

The second model incorporates regularization and dropout layers to reduce overfitting and improve generalization. The architecture is as follows:

| Layer Type   | Filters | Kernel Size | Activation | Pool Size | Dropout Rate | Regularization |
| ------------ | ------- | ----------- | ---------- | --------- | ------------ | -------------- |
| Conv2D       | 32      | (3, 3)      | ReLU       | -         | -            | L2(0.001)      |
| MaxPooling2D | -       | -           | -          | (2, 2)    | 0.4          | -              |
| BatchNormalisation | -       | -           | -          | -    | -          | -              |
| Conv2D       | 64      | (3, 3)      | ReLU       | -         | -            | L2(0.001)      |
| MaxPooling2D | -       | -           | -          | (2, 2)    | 0.4          | -              |
| BatchNormalisation | -       | -           | -          | (2, 2)    | 0.4          | -              |
| Flatten      | -       | -           | -          | -         | -            | -              |
| Dense        | 64      | -           | ReLU       | -         | 0.4          | L2(0.001)      |
| BatchNormalisation | -       | -           | -          | -   |   -    | -              |
| Dense        | 32      | -           | ReLU       | -         | -            | L2(0.001)      |
| Output       | -       | -           | Softmax    | -         | -            | -              |

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Precision, Recall
- **Callbacks**:
  - Early Stopping (patience=7)
  - Model Checkpoint (best model saved to `callbacks/model_checkpoint.keras`)

**Training Results**:

- **Final Training Accuracy**: 91.94%
- **Final Validation Accuracy**: 80.11%

## Regularization

### L2 vs. L Regularization

We applied **L2** regularization instead of **L1**. **L2** regularization reduces the overall magnitude of weights, helping to prevent overfitting without necessarily promoting sparsity. This approach was chosen to enhance generalization without focusing on feature selection.

## Optimizers

### Adam vs. Adamax and RMSprop

**Adam** vs. **Adamax** and **RMSprop**
We chose the **Adam optimizer** instead of **Adamax** or **RMSprop**. **Adam** combines momentum with adaptive learning rates, making it a versatile choice for a wide range of tasks. Here’s a brief comparison:

**Adam**: A widely-used optimizer that balances momentum and adaptive learning rates, suitable for most tasks.
**RMSprop**: Focuses on adaptive learning rates, which helps with non-stationary data, though it may converge more slowly.
**Adamax**: A variant of Adam based on the infinity norm, which can be advantageous with large gradients or outliers.

## Model Training and Callbacks

We implemented the following callbacks to enhance the model's training:

- **EarlyStopping**: Monitors `val_loss` and stops training when no improvement is seen after 20 epochs, restoring the best weights.

  ```python
  early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, mode='auto', verbose=1)
  ```

- **ModelCheckpoint**: Saves the best model based on validation performance.

  ```python
  check_point = ModelCheckpoint("callbacks/model_checkpoint.keras", save_best_only=True,  mode='auto', verbose=1)
  ```

- **ReduceLROnPlateau**: Reduces the learning rate by a factor of 0.2 if there’s no improvement in `val_loss` for 5 epochs, with a minimum learning rate of 0.0001.

  ```python
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)
  ```

## 3. Conclusion

- **Model 1** achieved higher training accuracy but struggled with generalization, as indicated by its lower validation accuracy.
- **Model 2** incorporated dropout and L2 regularization, which improved generalization performance, achieving a validation accuracy closer to the training accuracy.

Both models demonstrate the impact of regularization and optimization techniques on model performance, highlighting the balance between training accuracy and generalization. Model 2 is saved as `saved_models/model2.keras` and recommended for deployment.

### How to Run

To set up and run the project, follow these steps:

1. **Clone the repository**  
   Clone the repository to your local machine using the following commands:

   ```bash
   git clone https://github.com/AristideI/brain-tumor-detection-cnn-model
   cd your-repository
   ```

2. **Create a Virtual Environment**

It’s recommended to create a virtual environment to manage dependencies. You can use either `venv` or `conda`:

- **Using `venv`:**

```bash
  python3 -m venv venv
  source venv/bin/activate   # On macOS/Linux
  venv\Scripts\activate      # On Windows
```

- **Using conda:**

```bash
conda create --name your_env_name python=3.8
conda activate your_env_name
```

3. **Install Dependencies**

Install the required dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## run everything in `notebook.ipynb`
