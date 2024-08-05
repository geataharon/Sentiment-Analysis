# Sentiment Analysis of Amazon Product Reviews

## Project Description
This project, developed by Geat Aharon, applies sentiment analysis to Amazon product reviews to classify them as positive, neutral, or negative. Utilizing Python, spaCy, and SpacyTextBlob, the analysis focuses on extracting sentiments from textual feedback and aligning them with numerical ratings.

## Dataset Description
The dataset, named `amazon_product_reviews.csv`, contains various attributes including dates, IDs, URLs, manufacturer details, product ratings, and review texts. The analysis predominantly uses `reviews.rating` and `reviews.text` to assess sentiment.

## Preprocessing Steps
- **Data Loading:** Loaded using Pandas with `low_memory=False` to efficiently handle large datasets.
- **Sentiment Mapping:** Ratings are mapped to sentiments ('Positive', 'Neutral', 'Negative').
- **Removing Missing Values:** Rows without `reviews.text` are dropped to maintain data integrity.
- **Applying Sentiment Analysis:** Sentiments are analyzed using the `analyze_sentiment` function, which relies on spaCy and SpacyTextBlob.

## Model Evaluation
The sentiment analysis model demonstrates an accuracy of 82.45%, effectively matching sentiments with the review ratings. It also features capabilities to assess semantic similarities between reviews and perform detailed sentiment analysis on selected samples.

## Insights
### Strengths
- Simplicity and efficiency through the use of spaCy and SpacyTextBlob.
- Good accuracy rate facilitating reliable sentiment classification.
- Versatility in analyzing both sentiment and text similarities.

### Limitations
- Reliance on static polarity thresholds might miss nuanced emotional expressions.
- Model performance could vary with texts from different domains or styles.
- Dependence on the specific features of spaCy's language model.

## Installation
To set up this project locally, follow these steps:
```bash
git clone <repository-url>
cd <repository-directory>
pip install -r requirements.txt
