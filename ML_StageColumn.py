#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import RandomOverSampler
import chardet

# Configuration
#stop_words = 'english'
stopwords = list(nltk.corpus.stopwords.words('english'))

# Add your custom stopwords
custom_words = ['copyright', 'publication', 'abstract']
stopwords.extend(custom_words)
stopwords = [word.lower() for word in stopwords]
nfold = 5
ngram = 5  # Match article's 5-gram
opt_met = {'f1': 3, 'precision': 4, 'recall': 5}


def detect_encoding(file_path):
    """Robust encoding detection with fallback"""
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding'] if result['confidence'] > 0.7 else 'latin1'

def clean_text(text):
    """Handle all encoding issues"""
    if not isinstance(text, str):
        text = str(text, errors='replace')
    return text.encode('latin1', 'ignore').decode('latin1', errors='replace').strip()

def vectorize(txt, target, ngram, vectorizer=None, selector=None):
    """TF-IDF with feature selection (train/test modes)"""
    if vectorizer is None and selector is None:  # Training mode
        vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, ngram),
                                    max_df=0.5, stop_words=stopwords)
        txt_vec = vectorizer.fit_transform(txt)
        selector = SelectKBest(chi2, k='all')  # Increased features
        txt_vec = selector.fit_transform(txt_vec, target)
        return txt_vec, vectorizer, selector
    else:  # Test mode
        txt_vec = vectorizer.transform(txt)
        return selector.transform(txt_vec)

def assign_screening_stages(predictions, probabilities):
    """Assign screening stages based on ML predictions and probability scores"""
    # Initialize all as Stage 3
    stages = ['Stage 3 - Exclude without Review'] * len(predictions)
    
    # Stage 1: All predicted relevant articles
    stage1_indices = np.where(predictions == 1)[0]
    for idx in stage1_indices:
        stages[idx] = 'Stage 1 - Manual Review Required'
    
    # Stage 2: Top 500 highest probability articles from class 0 (predicted non-relevant)
    class0_indices = np.where(predictions == 0)[0]
    if len(class0_indices) > 0:
        # Get probabilities for class 0 articles only
        class0_probs = probabilities[class0_indices]
        # Sort by probability (highest first) and take top 500
        sorted_indices = np.argsort(-class0_probs)  # Negative for descending order
        top_500_indices = class0_indices[sorted_indices[:500]]
        
        for idx in top_500_indices:
            stages[idx] = 'Stage 2 - Insurance Review (Top 500)'
    
    return stages

def mlfunc(train_file, test_file, txt_col_train, target_col, 
          txt_col_test, opt_metric='recall', manual_encoding='latin1'):
    """Full optimized workflow for recall improvement"""
    
    # Load and clean training data
    try:
        df_train = pd.read_csv(train_file, encoding=manual_encoding,
                              engine='python', on_bad_lines='skip', dtype=str)
    except UnicodeDecodeError:
        df_train = pd.read_csv(train_file, encoding='ISO-8859-1',
                              engine='python', on_bad_lines='skip', dtype=str)
    
    df_train = df_train.dropna().sample(frac=1).reset_index(drop=True)
    txt_train = df_train.iloc[:, txt_col_train].apply(clean_text).tolist()
    target = df_train.iloc[:, target_col]
    
    # Handle target encoding
    n_classes = target.nunique()
    target = target.astype(np.float64) if n_classes == 2 else target.astype('category').cat.codes

    # Initialize classifier with class balancing
    base_clf = LinearSVC(random_state=45, class_weight='balanced')
    ##calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=5)
    calibrated_clf = CalibratedClassifierCV(base_clf, cv=5)
    
    # Model evaluation with recall focus
    results = []
    for ng in range(1, ngram+1):
        X_train, vec, sel = vectorize(txt_train, target, ng)
        
        # Optional: Uncomment for oversampling
        # ros = RandomOverSampler(random_state=45)
        # X_train, target = ros.fit_resample(X_train, target)
        
        scores = [
            np.mean(cross_val_score(calibrated_clf, X_train, target, cv=nfold, scoring='f1')),
            np.mean(cross_val_score(calibrated_clf, X_train, target, cv=nfold, scoring='precision')),
            np.mean(cross_val_score(calibrated_clf, X_train, target, cv=nfold, scoring='recall'))
        ]
        results.append(('LinearSVC', ng, len(txt_train), *scores))
    
    # Select best model by recall
    res = pd.DataFrame(results, columns=['Classifier', 'Ngram', 'n_Training', 'F1', 'Precision', 'Recall'])
    best_row = res.iloc[res['Recall'].idxmax()]
    best_ngram = int(best_row['Ngram'])  # Fix: Convert to int
    
    # Final training with best parameters
    X_train, final_vec, final_sel = vectorize(txt_train, target, best_ngram)
    calibrated_clf.fit(X_train, target)
    
    # Load and process test data
    df_test = pd.read_csv(test_file, encoding=manual_encoding,
                         engine='python', on_bad_lines='skip', dtype=str)
    txt_test = df_test.iloc[:, txt_col_test].apply(clean_text).tolist()
    X_test = vectorize(txt_test, None, best_ngram, final_vec, final_sel)
    
    # Generate predictions with adjusted threshold
    prob = calibrated_clf.predict_proba(X_test)[:, 1]  # Use calibrated probabilities
    pred = calibrated_clf.predict(X_test)
    
    # Format output
    df_test['Prediction'] = pred
    df_test['Probability'] = np.round(prob, 3)
    df_test['Screening_Stage'] = assign_screening_stages(pred, prob)  # Add screening stages
    
    return df_test, res  # Return only relevant predictions

# Usage remains the same as previous example
# Usage
if __name__ == "__main__":
    train_df, performance_df = mlfunc(
        train_file = "Sim2_Training.csv",
        test_file = "Sim2_Unclassified.csv",
        txt_col_train = 1,   # 0-based index
        target_col = 2,      
        txt_col_test = 1,    
        opt_metric = 'recall',
        manual_encoding = 'latin1'  # Force Latin-1 for Windows compatibility
    )
    
    train_df.to_csv("predictions_sim2_stage.csv", index=False, encoding='latin1')
    performance_df.to_csv("performance_metrics.csv", index=False, encoding='latin1')
    print("Success! Files saved with Latin-1 encoding.")
    
    # Print screening stage summary
    stage_counts = train_df['Screening_Stage'].value_counts()
    print(f"\nScreening Stage Distribution:")
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count} articles ({count/len(train_df)*100:.1f}%)")

