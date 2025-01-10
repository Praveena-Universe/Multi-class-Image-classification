# Multi-class-Image-classification

Data Source: https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification

Download the dataset to local machine

# README for Image Classification using SVM

## Project Overview
This project demonstrates the use of Support Vector Machines (SVM) for multi-class image classification. The primary focus is on comparing the effects of various preprocessing steps, such as balancing datasets, data augmentation, and normalization, on the performance of SVM models. The dataset includes images categorized into four classes: `water`, `green_area`, `desert`, and `cloudy`.

## Project Structure
```
project/
│
├── data/
│   ├── water/
│   ├── green_area/
│   ├── desert/
│   ├── cloudy/
│
├── scripts/
│   ├── svm_without_preprocessing.py
│   ├── svm_with_balanced_sampling.py
│   ├── svm_with_preprocessing.py
│   ├── data_augmentation.py
│
└── README.md
```

### Dataset
The dataset is stored in the `data/` folder. Each subfolder represents a class and contains corresponding image files.

### Key Scripts
### Key Note: Conver the ipynb files to py before running scripts for a complete flow of each file
1. **svm_without_preprocessing.py**  
   Implements SVM without any preprocessing or balancing. It directly uses the raw grayscale and resized images for training and testing.  

2. **svm_with_balanced_sampling.py**  
   Implements SVM with balanced sampling but without additional preprocessing steps. Balancing ensures each class has an equal representation in the dataset.  

3. **svm_with_preprocessing.py**  
   Includes full preprocessing steps:
   - Label encoding of target variables.
   - Resizing all images to `(64, 64)` dimensions.
   - Scaling image data using `StandardScaler` for normalization.
   - Balanced sampling for dataset representation.

4. **data_augmentation.py**  
   Uses the `ImageDataGenerator` from Keras to apply data augmentation techniques for oversampling classes with fewer images. Augmentation includes:
   - Rotation
   - Shearing
   - Zooming
   - Horizontal flipping

## Steps to Run the Project
### Prerequisites
- Python 3.x
- Required libraries:
  ```
  pip install numpy scikit-learn opencv-python keras tensorflow
  ```

### Running the Scripts
1. **Prepare the Dataset**
   - Place your image dataset in the `data/` directory.
   - Ensure the folder structure aligns with class names.

2. **SVM Without Preprocessing**
   ```bash
   python scripts/svm_without_preprocessing.py
   ```
   Outputs:
   - Accuracy
   - Precision, Recall, F1-Score

3. **SVM With Balanced Sampling**
   ```bash
   python scripts/svm_with_balanced_sampling.py
   ```
   Outputs:
   - Improved classification metrics due to balanced data representation.

4. **SVM With Preprocessing**
   ```bash
   python scripts/svm_with_preprocessing.py
   ```
   Outputs:
   - Further enhanced metrics due to scaling and normalization.

5. **Data Augmentation**
   ```bash
   python scripts/data_augmentation.py
   ```
   Outputs:
   - Augmented images saved in the corresponding class folders.

## Results and Observations
### Evaluation Metrics
- **Accuracy**: Proportion of correct predictions.
- **Precision**: Fraction of relevant instances among retrieved instances.
- **Recall**: Fraction of relevant instances retrieved.
- **F1-Score**: Harmonic mean of precision and recall.

| Model                    | Accuracy | Precision | Recall | F1-Score |
|--------------------------|----------|-----------|--------|----------|
| Without Preprocessing    | 0.647    | 0.672     | 0.618  | 0.609    |
| Balanced Sampling        | 0.650    | 0.696     | 0.638  | 0.627    |
| With Preprocessing       | 0.796    | 0.812     | 0.793  | 0.786    |

### Conclusion
- Balancing the dataset improves overall performance.
- Preprocessing and normalization significantly enhance SVM's ability to classify images.

## Future Enhancements
- Experiment with different SVM kernels (e.g., `rbf`, `poly`).
- Integrate more advanced machine learning models such as CNNs for comparison.
- Automate the preprocessing pipeline for larger datasets.

## Author
Praveena Silmala  
