

# <div align="center">Parkinson's disease prediction </div>
<div align="center"><img src="https://github.com/Pradnya1208/Detecting-Parkinsons-Disease/blob/main/output/intro.gif?raw=true" width="70%"></div>



## Overview:

#### What is Parkinson’s Disease?
Parkinson's disease (PD), or simply Parkinson's is a long-term degenerative disorder of the central nervous system that mainly affects the motor system. The symptoms usually emerge slowly and, as the disease worsens, non-motor symptoms become more common. The most obvious early symptoms are tremor, rigidity, slowness of movement, and difficulty with walking,but cognitive and behavioral problems may also occur. Parkinson's disease dementia becomes common in the advanced stages of the disease. Depression and anxiety are also common, occurring in more than a third of people with PD. Other symptoms include sensory, sleep, and emotional problems. The main motor symptoms are collectively called "parkinsonism", or a "parkinsonian syndrome.

<img src="https://github.com/Pradnya1208/Parkinsons-disease-prediction-using-CNN/blob/main/output/overview.PNG?raw=true">
<br>
While Parkinson’s disease cannot be cured, early detection along with proper medication can significantly improve symptoms and quality of life, making it an important topic for research especially in the creation of new diagnostic tools.
<br>

#### What is the spiral drawing study?
A 2017 study by Zham et al. found that it was possible to detect Parkinson’s by asking the patient to draw a spiral and then track:

- Speed of drawing
- Pen pressure
The researchers found that the drawing speed was slower and the pen pressure lower among Parkinson’s patients — this was especially pronounced for patients with a more acute/advanced forms of the disease.

We’ll be leveraging the fact that two of the most common Parkinson’s symptoms include tremors and muscle rigidity which directly impact the visual appearance of a hand drawn spiral and wave.


The variation in visual appearance will enable us to train a computer vision + machine learning algorithm to automatically detect Parkinson’s disease.
## Dataset:
[Parkinson's Drawings](https://www.kaggle.com/kmader/parkinsons-drawings)

The dataset we’ll be using here today was curated by Adriano de Oliveira Andrade and Joao Paulo Folado from the NIATS of Federal University of Uberlândia.

The dataset itself consists of images and is pre-split into a training set and a testing set, consisting of:

- Spiral: training, and testing

- Wave: training, and testing

<img src="https://github.com/Pradnya1208/Parkinsons-disease-prediction-using-CNN/blob/main/output/dataset.PNG?raw=true">


## Implementation:

## Implementation
### Libraries Used
- **Data Processing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib
- **Deep Learning**: TensorFlow, Keras

### Machine Learning Models
- **Gait Analysis**: Applied models include Logistic Regression, Decision Trees, Random Forest, SVM (Linear, RBF, Poly Kernels), k-NN, and XGBoost.
- **Spiral Drawing Analysis**: A CNN classifier was developed with Conv2D, MaxPooling2D, Flatten, and Dense layers. The Adam optimizer was used with binary cross-entropy loss, and callbacks were implemented for training.

### Best Performing Model
- **Random Forest Classifier**:
  - Accuracy: 0.996
  - F1 Score: 0.961
  - R2 Score: 0.862

## Learnings
- Efficient data preprocessing and compression for large datasets.
- Comparison of various ML models for classification tasks.
- Implementation and optimization of CNN for image-based diagnosis.

## Dataset Details
- **Parkinson’s Disease Dataset**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data)
- **Parkinson’s Drawings**: Training and testing sets for spirals and wave drawings.


## Data Exploration:
#### Spirals and waves drawn by healthy people:
<img src="https://github.com/Pradnya1208/Parkinsons-disease-prediction-using-CNN/blob/main/output/healthy.PNG?raw=true" width="45%"> <img src="https://github.com/Pradnya1208/Parkinsons-disease-prediction-using-CNN/blob/main/output/wave-healthy.PNG?raw=true" width="45%">

#### Spirals and waves drawn by people with Parkinson:
<img src="https://github.com/Pradnya1208/Parkinsons-disease-prediction-using-CNN/blob/main/output/parkinson.PNG?raw=true" width= "45%"> <img src="https://github.com/Pradnya1208/Parkinsons-disease-prediction-using-CNN/blob/main/output/wave-parkinson.PNG?raw=true" width= "45%">

## Model training and evaluation:

```
train_datagen = ImageDataGenerator(rescale = 1./255, 
                                  shear_range = 0.2, 
                                  zoom_range = 0.2, 
                                  horizontal_flip = True)
```
```
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
```
```
classifier=Sequential()
classifier.add(Conv2D(32,(3,3),input_shape=(128, 128, 3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(32,(3,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())
classifier.add(Dense(activation='relu',units=128))
classifier.add(Dense(activation='sigmoid',units=1))
```
```
classifier.compile(loss='binary_crossentropy',
              optimizer = Adam(learning_rate=0.001),
              metrics=['accuracy'])
```

```
history = classifier.fit(
        spiral_train_generator,
        steps_per_epoch=spiral_train_generator.n//spiral_train_generator.batch_size,
        epochs=48,
        validation_data=spiral_test_generator,
        validation_steps=spiral_test_generator.n//spiral_test_generator.batch_size,
        callbacks=callbacks_list)
```
<img src="https://github.com/Pradnya1208/Parkinsons-disease-prediction-using-CNN/blob/main/output/accuracy.PNG?raw=true">

### Learnings:
`CNN Classifier` `Callbacks`






## References:
[XGBoost](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/) <br>
[Callbacks](https://keras.io/api/callbacks/)
