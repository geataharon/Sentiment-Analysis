# Import necessary libraries
import pandas as pd
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import random

# Configure pandas display options for better readability
pd.set_option('expand_frame_repr', False)  # Prevent wrapping to new line if too many columns
pd.set_option('display.max_rows', 5000)  # Set maximum number of rows to display
pd.set_option('display.max_columns', 5000)  # Set maximum number of columns to display
pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)

# Load the spaCy English language model (I specifically chose a larger model than the _sm that was requested)
nlp = spacy.load("en_core_web_md")

# Add SpacyTextBlob to the pipeline
nlp.add_pipe('spacytextblob')

# Function to preprocess text and perform sentiment analysis
def analyze_sentiment(review):
    doc = nlp(review)
    return "Positive" if doc._.polarity > 0.1 else "Negative" if doc._.polarity < -0.1 else "Neutral"

# Load dataset with low_memory=False option
df = pd.read_csv("amazon_product_reviews.csv", low_memory=False)

# Map numerical ratings to sentiment labels
df['original_sentiment'] = df['reviews.rating'].apply(lambda x: 'Positive' if x > 3 else ('Neutral' if x == 3 else 'Negative'))

# Remove missing values in 'reviews.text'
clean_reviews = df.dropna(subset=['reviews.text']).copy()

# Apply sentiment analysis
clean_reviews.loc[:, 'sentiment_predict'] = clean_reviews['reviews.text'].apply(analyze_sentiment)

# Compare analyzed sentiment with original sentiment derived from ratings
correct_predictions = (clean_reviews['original_sentiment'] == clean_reviews['sentiment_predict']).sum()
total_predictions = clean_reviews.shape[0]
accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy:.4f}")

# Randomly choose two reviews and compare their similarity
review_indices = random.sample(range(0, len(clean_reviews)), 2)
review1 = clean_reviews.iloc[review_indices[0]]['reviews.text']
review2 = clean_reviews.iloc[review_indices[1]]['reviews.text']
doc1 = nlp(review1)
doc2 = nlp(review2)
print(f"Review 1: {review1}")
print(f"Review 2: {review2}")
print(f"Similarity: {doc1.similarity(doc2):.4f}")

# Testing model on a sample of product reviews
sample_reviews = clean_reviews['reviews.text'].sample(5).tolist()
for i, review in enumerate(sample_reviews, 1):
    sentiment = analyze_sentiment(review)
    print(f"Sample Review {i}: {review[:100]}... -> Predicted Sentiment: {sentiment}")
