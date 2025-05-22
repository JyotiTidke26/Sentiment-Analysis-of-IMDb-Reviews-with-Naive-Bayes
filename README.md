# 🎯 Sentiment Analysis of IMDb Reviews with Naive Bayes

## 📌 Project Overview
This project implements a sentiment analysis pipeline using the **Naive Bayes** classifier. The dataset consists of text reviews (positive and negative sentiments) which are preprocessed and vectorized before training a model. The goal of the project is to classify whether a given review expresses a positive or negative sentiment.

## 🎯 Objective
The objective of this project is to predict the sentiment of text reviews (positive or negative) using a machine learning model. We specifically use the **Multinomial Naive Bayes** classifier, a simple yet effective model for text classification tasks.

## 💡 Why This Project?
This project was built to:
- Explore and implement sentiment analysis techniques using machine learning.
- Showcase the application of the **Naive Bayes** algorithm in natural language processing (NLP).
- Demonstrate the use of text preprocessing, vectorization, and evaluation metrics like confusion matrix and classification report.

## 🔑 Key Steps

### 📦 1. Importing Required Libraries
- **scikit-learn**: Used for building the Naive Bayes model, vectorization, and model evaluation.
- **Pandas**: Used for data manipulation.
- **Matplotlib/Seaborn**: Used for visualizing results and performance metrics.

### 🗂️ 2. Dataset Loading & Splitting
- The dataset consists of labeled reviews (positive or negative).
- It is split into training and testing sets for model evaluation.

### 🧼 3. Preprocessing
- **Text Cleaning**: Removal of punctuation, stop words, and non-relevant characters.
- **Vectorization**: Text is converted into numerical data using techniques like **CountVectorizer** or **TfidfVectorizer**.

### 🧪 4. Model Building & Training
- **Multinomial Naive Bayes**: The model is trained on the preprocessed and vectorized text data.
- Hyperparameters, such as `alpha` (Laplace smoothing), are tuned using **GridSearchCV** for optimal performance.

### 🚀 5. Model Evaluation
- **Classification Report**: Evaluates precision, recall, f1-score for each class.
- **Confusion Matrix**: Displays the true vs predicted classifications.
- 
### 🚀 6. Hyperparameter Tuning
Hyperparameter tuning is a critical part of improving model performance. In this project, we used **GridSearchCV** to search for the best hyperparameters for the **Multinomial Naive Bayes** model. The primary hyperparameter we tuned was `alpha`, which is used for Laplace smoothing. 

#### Hyperparameter Tuning with GridSearchCV:
- **Alpha**: This hyperparameter controls the smoothing applied during the estimation of probabilities. We experimented with different values of `alpha` to determine the one that minimizes the error on the validation set.
  
### 🏆 6. Results
The model is evaluated using a **classification report** and a **confusion matrix** to assess its accuracy and performance. The model achieves good accuracy with minimal overfitting.

## 📈 Model Performance
| Metric              | Value   |
|---------------------|---------|
| Accuracy            | 85%     |
| Precision (Positive)| 0.85    |
| Recall (Positive)   | 0.86    |
| F1-Score (Positive) | 0.85    |
| Precision (Negative)| 0.86    |
| Recall (Negative)   | 0.84    |
| F1-Score (Negative) | 0.85    |

## 📁 Files Included
- **Sentiment_Analysis.ipynb**: The Jupyter notebook that includes the complete workflow, including data loading, preprocessing, model training, and evaluation.

## 💡 Recommendations for Further Improvement
Here are a few ways this project can be extended and improved:
- **Experiment with Other Models**: Try other machine learning models like **Logistic Regression**, **SVM**, or **Random Forest** for comparison.
- **Deep Learning**: Experiment with deep learning models such as **LSTM** or **BERT** for sentiment analysis on larger datasets.
- **Data Augmentation**: Use techniques like back-translation or synonym replacement to augment the text data.

## 📦 Requirements
- **scikit-learn**: 0.24.x
- **pandas**: 1.1.x
- **matplotlib**: 3.3.x
- **seaborn**: 0.11.x

## 🧠 Author Note
Feel free to explore and experiment with this project! For any questions or suggestions, don’t hesitate to reach out.

---
*Feel free to fork, clone, or open an issue if you encounter any problems or have suggestions for improvements.*
