# Sentiment Analysis NLP project
- In this project we make binary and multi-class sentiment analysis.
- The reviews are in many languages, so we have to detect first which language is used then decide wherever review is positive or negative.
- The training dataset is ```imdb_reviews.csv``` movie reviews. But it is only available for binary sentiment analysis. We have to make multi-class analysis.

![workflow-diagram.png](diagrams/workflow-diagram.png)

## Prerequisites
1. Make sure to have python installed.
2. Clone this repository.
3. Create a ```.venv```. It would be perfect to configurate your IDE to compiler to you ```.venv```.
4. Activate your ```.venv```
5. Install ```jupyter notebooks```.
6. Enjoy coding!

## The process
### Text preprocessing
At this step we clean, translate, tokenize, lemmatize the reviews if needed.

### Classification

#### Aproach
We classified IMDB movie reviews as positive (1) or negative (0). Preprocessed text using TF-IDF vectorization. 

Trained two models:
1. Logistic Regression
2. Random Forest Classifier

#### Model Performance Summary
| Metric         | Logistic Regression | Random Forest |
|----------------|---------------------|----------------|
| **Accuracy**   | **0.86**            | 0.82           |
| **Precision (0)** | 0.89              | 0.82           |
| **Recall (0)**    | 0.83              | 0.83           |
| **Precision (1)** | 0.84              | 0.82           |
| **Recall (1)**    | 0.89              | 0.81           |
| **F1-score (avg)**| 0.86              | 0.82           |
| **ROC AUC**       | **0.937**         | 0.904          |

#### Confusion Matrix:
| Type                | Predicted Negative | Predicted Positive |
|---------------------|---|---|
| **Actual Negative** | 83% (Correct)       | 17% (False Positive) |
| **Actual Positive** | 11% (False Negative) | 89% (Correct)        |

#### Conclusion
Overall, Logistic Regression outperforms Random Forest across all major metrics. It achieves higher precision, recall, and ROC AUC, indicating better generalization.
The confusion matrix supports this with fewer false predictions.

### Sentiment Prediction Model
We built a binary sentiment classification model to predict whether a cleaned text review is positive or negative.

#### Data:
1. Input: cleaned_review (preprocessed text)
2. Label: label (0 = negative, 1 = positive)

#### Model Architecture:

- Embedding Layer (dim=100)
- 2-layer Bidirectional LSTM (hidden_dim=256)
- Fully Connected Layer 
- Dropout (p=0.5)
- Output: 1 sigmoid-activated unit for binary classification

#### Training:
- Loss: Binary Cross-Entropy with Logits 
- Optimizer: Adam 
- Epochs: 10 
- Best model saved based on ```validation loss``` 

#### Evaluation:
- Test Accuracy: (insert value from final output)
- Model loaded from ```best_model.pt``` 

