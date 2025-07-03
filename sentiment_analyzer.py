"""
sentiment_analyzer.py

This script implements a machine learning pipeline for sentiment analysis of product reviews.
It is specifically designed to process the 'Amazon Fine Food Reviews' dataset from Kaggle,
classifying reviews as 'positive' or 'negative'.

The script covers:
1. Data Loading and Preprocessing: Handling raw text, cleaning, stopword removal, and stemming.
2. Feature Extraction: Converting text to numerical data using TF-IDF.
3. Model Training and Evaluation: Using Logistic Regression to classify sentiment,
   and providing performance metrics (accuracy, classification report, confusion matrix).
4. Model Persistence: Saving and loading the trained model for new predictions.

This serves as a robust example of implementing an AI solution for text classification.
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os # For creating directories

# --- Download NLTK data (only need to do this once on your machine) ---
# This ensures that stopwords are available for text preprocessing.
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords. This is a one-time download and requires internet access...")
    nltk.download('stopwords')
    print("NLTK stopwords download complete.")

# --- Ensure 'models' directory exists ---
# The trained model and vectorizer will be saved here.
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models/' directory.")

# --- 1. Data Loading and Preprocessing ---

def load_and_prepare_data(filepath):
    """
    Loads data from the 'Reviews.csv' (Amazon Fine Food Reviews) file.
    Maps 'Score' column to 'sentiment' labels and filters out 'neutral' reviews.
    """
    print(f"\n--- Loading data from {filepath} ---")
    # Read only necessary columns to save memory for large datasets
    df = pd.read_csv(filepath, usecols=['Text', 'Score'])
    
    # Rename 'Text' column to 'review' for consistency with our previous structure
    df.rename(columns={'Text': 'review'}, inplace=True)

    # Convert 'Score' to sentiment labels
    def score_to_sentiment(score):
        if score > 3:
            return 'positive'
        elif score < 3:
            return 'negative'
        else:
            return 'neutral'
            
    df['sentiment'] = df['Score'].apply(score_to_sentiment)
    
    # Drop the original 'Score' column
    df.drop('Score', axis=1, inplace=True)

    # Important: For this binary classification task, we remove 'neutral' reviews.
    # This aligns with the video narrative of focusing on actionable positive/negative feedback.
    original_rows = len(df)
    df = df[df['sentiment'] != 'neutral'].copy()
    print(f"Data loaded. Removed {original_rows - len(df)} 'neutral' reviews. Remaining {len(df)} actionable reviews.")
    
    # Basic check for empty dataframe after filtering
    if df.empty:
        raise ValueError("No positive or negative reviews found after filtering 'neutral'. Check your dataset.")

    # Remove rows with NaN in 'review' column if any
    df.dropna(subset=['review'], inplace=True)
    print(f"Removed reviews with missing text. Final count: {len(df)} reviews.")

    return df

def preprocess_text(text):
    """
    Cleans and preprocesses a single text string for machine learning.
    Steps: Lowercasing, removing non-alphabetic characters, stopword removal, and stemming.
    """
    if not isinstance(text, str):
        return "" # Handle non-string input, though dropna should prevent this for 'review'
        
    # 1. Lowercase the text
    text = text.lower()
    
    # 2. Remove punctuation and numbers (keeping only letters and spaces)
    # Using a compiled regex for slight performance improvement on large data
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. Tokenize the text (split into words)
    tokens = text.split()
    
    # 4. Remove stopwords (common words like 'the', 'is', 'in' that add little meaning)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # 5. Stemming (reduce words to their root form, e.g., 'running' -> 'run', 'amazed' -> 'amaz')
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # 6. Join tokens back into a single string
    return ' '.join(tokens)

# --- 2. Model Training and Evaluation ---

def train_model(df):
    """
    Trains the sentiment analysis model using the provided DataFrame.
    Includes text preprocessing, TF-IDF vectorization, data splitting,
    Logistic Regression training, model evaluation, and saving.
    """
    print("\n--- Starting model training process ---")

    # Apply the preprocessing function to all reviews
    print("Step 1/5: Preprocessing text data (this may take a few minutes for large datasets)...")
    df['processed_review'] = df['review'].apply(preprocess_text)
    
    # Filter out empty processed reviews that might result from aggressive preprocessing
    df = df[df['processed_review'].str.strip() != ''].copy()
    print(f"Filtered out empty processed reviews. Remaining: {len(df)} reviews.")

    # --- Feature Extraction (TF-IDF Vectorization) ---
    # Converts text data into numerical feature vectors.
    # TF-IDF weighs words by their frequency in a document and rarity across all documents.
    print("Step 2/5: Performing TF-IDF feature extraction...")
    # Increased max_features for a larger dataset to capture more vocabulary
    vectorizer = TfidfVectorizer(max_features=10000) # Use top 10,000 most frequent/important words
    X = vectorizer.fit_transform(df['processed_review'])
    
    # Convert sentiment labels to numerical format: 'positive' -> 1, 'negative' -> 0
    y = df['sentiment'].apply(lambda s: 1 if s == 'positive' else 0)

    # --- Split Data ---
    # Divide data into training (80%) and testing (20%) sets to evaluate model performance
    print("Step 3/5: Splitting data into training and testing sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train Logistic Regression Model ---
    # Logistic Regression is a good baseline classifier for binary classification.
    print("Step 4/5: Training the Logistic Regression model (this may take a few minutes)...")
    # Increased max_iter for convergence on larger datasets
    model = LogisticRegression(max_iter=2000, solver='liblinear') # 'liblinear' often works well for large datasets
    model.fit(X_train, y_train)

    # --- Evaluate the Model ---
    print("Step 5/5: Evaluating the model performance...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {accuracy:.4f}")
    
    print("\nClassification Report (Precision, Recall, F1-Score):")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    # --- Save the Model and Vectorizer ---
    # Save trained model and vectorizer for future predictions without retraining
    print("\n--- Saving the trained model and TF-IDF vectorizer ---")
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved successfully in the 'models/' directory.")

    # --- Visualize Confusion Matrix ---
    # Helps visualize where the model is making errors (false positives/negatives)
    print("Generating Confusion Matrix plot. Close the plot to continue...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Actual Sentiment')
    plt.title('Confusion Matrix for Sentiment Analysis')
    plt.show()

# --- 3. Prediction on New Data ---

def predict_sentiment(review_text):
    """
    Predicts the sentiment of a new, single review text using the saved model.
    """
    print("\n--- Making a new prediction ---")
    
    try:
        # Load the pre-trained model and vectorizer
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    except FileNotFoundError:
        print("Error: Model or vectorizer files not found in 'models/' directory.")
        print("Please run the script's training phase first to generate these files.")
        return None, None

    # Preprocess the new review text using the same function as during training
    processed_text = preprocess_text(review_text)
    
    # Vectorize the preprocessed text using the *fitted* vectorizer
    # .transform() expects an iterable, so wrap in a list
    vectorized_text = vectorizer.transform([processed_text])
    
    # Predict the sentiment (0 for negative, 1 for positive)
    prediction = model.predict(vectorized_text)
    
    # Predict the probability of each class
    probability = model.predict_proba(vectorized_text)
    
    sentiment_label = 'Positive' if prediction[0] == 1 else 'Negative'
    # Confidence for the predicted class
    confidence = probability[0][prediction[0]]
    
    print(f"Review: '{review_text}'")
    print(f"Predicted Sentiment: {sentiment_label} (Confidence: {confidence:.2f})")
    return sentiment_label, confidence


# --- Main execution block ---
if __name__ == "__main__":
    # Define the path to the Kaggle dataset
    kaggle_data_path = 'data/Reviews.csv'
    
    # Step 1: Load and prepare data, then train and evaluate the model.
    # This step will save 'sentiment_model.pkl' and 'tfidf_vectorizer.pkl' to the 'models/' folder.
    try:
        data_df = load_and_prepare_data(kaggle_data_path)
        if not data_df.empty:
            train_model(data_df)
        else:
            print("No valid data for training after processing. Please check the dataset path and content.")
    except FileNotFoundError:
        print(f"Error: Dataset not found at {kaggle_data_path}. Please download 'Reviews.csv' from Kaggle and place it in the 'data/' folder.")
    except ValueError as e:
        print(f"Data preparation error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")

    # Step 2: Use the trained model to make predictions on new, unseen reviews.
    # This part assumes the model files were successfully saved in Step 1.
    print("\n--- Demonstrating predictions on new product review examples ---")
    predict_sentiment("This coffee tastes amazing, best purchase ever!")
    predict_sentiment("The packaging was damaged and the product arrived broken. Very disappointed.")
    predict_sentiment("Good value for money, but the flavor is a bit bland.")
    predict_sentiment("Absolutely love this product. Will buy again.")
    predict_sentiment("Worst purchase of the year. Completely inedible.")