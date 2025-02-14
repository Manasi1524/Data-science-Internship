# Step 1: Import necessary libraries
import pandas as pd  # To load and handle the dataset
from sklearn.model_selection import train_test_split  # To split the dataset into train and test sets
from sklearn.feature_extraction.text import CountVectorizer  # To convert text into numerical vectors
from sklearn.preprocessing import LabelEncoder  # To convert text labels (spam/ham) into numeric values
from sklearn.naive_bayes import MultinomialNB  # To use Naive Bayes classification
from sklearn.metrics import accuracy_score, classification_report  # For evaluating model performance

# Step 2: Load the dataset
# Ensure the dataset is located in the same directory or provide the full path to the CSV file
df = pd.read_csv('spam.csv', encoding='latin-1')  # Load the dataset from CSV

# Step 3: Preprocess the dataset
# We select only relevant columns and rename them for clarity
df = df[['v1', 'v2']]  # Extract the label column (v1) and the message column (v2)
df.columns = ['label', 'message']  # Rename the columns to 'label' and 'message'

# Step 4: Encode labels (ham = 0, spam = 1)
label_encoder = LabelEncoder()  # Initialize the LabelEncoder to convert text labels to numbers
df['label'] = label_encoder.fit_transform(df['label'])  # Convert 'ham' to 0 and 'spam' to 1

# Step 5: Split the dataset into training and testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 6: Vectorize the text data (convert text to numerical format)
# Convert messages into a format the model can work with (bag of words)
vectorizer = CountVectorizer(stop_words='english')  # Use CountVectorizer to convert text to vectors (ignore common English words)
X_train = vectorizer.fit_transform(X_train)  # Fit and transform the training data
X_test = vectorizer.transform(X_test)  # Transform the test data

# Step 7: Initialize the Naive Bayes classifier
model = MultinomialNB()  # Initialize the Naive Bayes model

# Step 8: Train the model on the training data
model.fit(X_train, y_train)  # Train the model using the training data

# Step 9: Make predictions on the test set
y_pred = model.predict(X_test)  # Use the trained model to predict on the test data

# Step 10: Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
print(f"Accuracy: {accuracy}")  # Print the accuracy
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")  # Print precision, recall, f1-score

# Step 11: Define a function to predict if a new message is spam or not
def predict_spam(message):
    message_vect = vectorizer.transform([message])  # Convert the new message into a vector
    prediction = model.predict(message_vect)  # Predict if the message is spam (1) or not spam (0)
    return "Spam" if prediction == 1 else "Not Spam"  # Return the prediction result

# Example usage:
test_message = "Congratulations! You've won a free ticket to the Bahamas!"  # A sample message
print(predict_spam(test_message))  # Predict and print whether it's spam or not
