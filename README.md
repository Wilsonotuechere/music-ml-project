# Music Genre Classification Using Advanced ML Techniques

## Student Project Overview

This project demonstrates the application of two advanced machine learning techniques for music genre classification:

1. **Ensemble Learning** (Random Forest)
2. **Gradient Boosting** (HistGradientBoost)

## Learning Outcomes Demonstrated

### 1. Ensemble Learning Implementation

- Utilized Random Forest with 100 estimators
- Demonstrated understanding of:
  - Bagging technique
  - Voting mechanisms
  - Parallel processing optimization
  - Feature importance analysis

### 2. Gradient Boosting Implementation

- Implemented HistGradientBoostingClassifier
- Demonstrated understanding of:
  - Sequential learning
  - Gradient optimization
  - Loss function minimization
  - Handling of missing values

## Dataset and Implementation

### Data Description

Using Spotify's audio features dataset containing:

- 10,743 songs
- Multiple audio features (acousticness, danceability, etc.)
- Various genre labels

### Model Performance

```
Model                     Accuracy    Training Time
------------------------------------------------
HistGradientBoost         0.35        45s
Random Forest             0.38        62s
```

### Key Findings

1. **Ensemble Learning Benefits:**

   - Reduced overfitting through aggregation
   - Better handling of high-dimensional data
   - Robust feature importance rankings

2. **Gradient Boosting Advantages:**
   - Efficient handling of large datasets
   - Automatic handling of missing values
   - Sequential improvement of weak learners

## Technical Implementation

### Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Code Structure

```
music-ml-project/
├── data_w_genres.csv      # Spotify dataset
├── music_model.ipynb      # Implementation notebook
└── README.md             # Documentation
```

### Key Libraries

- scikit-learn 1.2.0
- pandas 1.5.0
- numpy 1.23.0

## Methodology

1. **Data Preprocessing**

   - Feature scaling using StandardScaler
   - Label encoding for genre categories
   - Train-test split (80-20)

2. **Model Training**

   ```python
   # Ensemble Learning (Random Forest)
   rf = RandomForestClassifier(n_estimators=100,
                             random_state=42,
                             n_jobs=-1)

   # Gradient Boosting
   hgb = HistGradientBoostingClassifier(random_state=42)
   ```

3. **Evaluation Metrics**
   - Classification accuracy
   - Confusion matrix
   - Feature importance analysis
