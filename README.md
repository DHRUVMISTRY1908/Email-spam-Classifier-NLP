# Email Spam Classifier using NLP



## Introduction

The Email Spam Classifier is an NLP (Natural Language Processing) project designed to distinguish between spam and legitimate emails based on their textual content. By employing various machine learning models and NLP techniques, the classifier aims to efficiently identify and filter out unwanted and potentially harmful spam emails. The project aims to enhance email management by providing accurate spam detection and real-time recommendations, ensuring users can focus on essential emails while keeping their inboxes clutter-free and secure. The interactive web interface and visualizations make the classifier user-friendly and informative, creating a seamless and productive email experience.
## Features

- **Spam Detection:** The core functionality of the classifier lies in its ability to effectively detect spam emails. By analyzing the content and structure of emails, it assigns a probability score, indicating the likelihood of an email being spam.

- **Preprocessing:** The text data from emails undergoes a series of preprocessing steps. These include tokenization, which breaks the text into individual words or tokens; stopword removal, where common words with little meaning are filtered out; stemming, which reduces words to their root form; and lowercasing, to ensure uniformity.

- **Multiple Machine Learning Models:** To achieve high accuracy, the Email Spam Classifier employs a variety of machine learning models. These include Gaussian Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, Support Vector Machine, Decision Tree, k-Nearest Neighbors, Random Forest, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost classifiers.

- **Model Evaluation:** To assess the performance of each model, we utilize evaluation metrics such as accuracy, precision, recall, and F1-score. This enables us to identify the best-performing model and fine-tune its parameters for optimal results.

- **Interactive Web Interface:** We have created an intuitive web interface where users can input email content and receive the spam probability score instantly. The clean and user-friendly design ensures a seamless user experience.

- **Real-time Recommendations:** As users interact with the classifier and provide feedback on email classifications, the system continuously learns and adapts, improving the accuracy of future predictions.

- **Data Visualization:** To provide insights into the classifier's performance, we utilize Matplotlib and Seaborn to create informative visualizations, such as confusion matrices and precision-recall curves.

The Email Spam Classifier aims to enhance email security and productivity by helping users efficiently manage their email communication, ensuring important messages receive attention while spam and potentially harmful content are promptly filtered out.

## Usage

To run the Email Spam Classifier locally, follow these steps:

1. Preprocess the data using various text preprocessing techniques, such as removing stopwords, tokenization, and stemming.
2. Create Bag of Words (BoW) or TF-IDF representations of the text data.
3. Split the dataset into training and testing sets.
4. Train different machine learning models using the preprocessed data.
5. Evaluate the performance of each model using appropriate metrics such as accuracy, precision, recall, and F1-score.
6. Select the best-performing model and use it to make predictions on new email data.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn


## Contributing

Contributions to the Email Spam Classifier project are welcome! If you have ideas for improvements or find any issues, feel free to open an issue or submit a pull request.


Thank you for using the Email Spam Classifier! We hope this project helps you in identifying and filtering out spam emails effectively. Happy email classification!
