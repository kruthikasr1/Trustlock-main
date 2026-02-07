# Import Libraries
# Importing Numpy & Pandas for data processing & data wrangling
import numpy as np
import pandas as pd

# Importing tools for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import evaluation metric libraries
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report

# Word Cloud library
from wordcloud import WordCloud, STOPWORDS

# Library used for data preprocessing
from sklearn.feature_extraction.text import CountVectorizer

# Import model selection libraries
from sklearn.model_selection import train_test_split

# Library used for ML Model implementation
from sklearn.naive_bayes import MultinomialNB

# Importing the Pipeline class from scikit-learn
from sklearn.pipeline import Pipeline

# Import joblib for saving model
import joblib

# Import library for saving plots
import os

# Import json for saving metrics
import json

# Library used for ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("https://raw.githubusercontent.com/Apaulgithub/oibsip_taskno4/main/spam.csv", encoding='ISO-8859-1')
print("=" * 70)
print("DATA LOADING AND PREPROCESSING")
print("=" * 70)
print("Number of rows are: ", df.shape[0])
print("Number of columns are: ", df.shape[1])

# Check for duplicates
dup = df.duplicated().sum()
print(f'Number of duplicated rows are: {dup}')

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Data cleaning
print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)
df.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
print("Renamed columns: Category, Message")
df.drop(columns={'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'}, inplace=True)
print("Removed unnecessary columns")

# Create binary target variable
df['Spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
print(f"\nClass distribution:")
print(df['Category'].value_counts())
print(f"\nSpam (1): {(df['Spam'] == 1).sum()} | Ham (0): {(df['Spam'] == 0).sum()}")

# Splitting the data to train and test
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.Spam, test_size=0.25, random_state=42, stratify=df.Spam)
print(f"\n" + "=" * 70)
print("TRAIN-TEST SPLIT")
print("=" * 70)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print(f"Training spam percentage: {(y_train.sum()/len(y_train))*100:.2f}%")
print(f"Testing spam percentage: {(y_test.sum()/len(y_test))*100:.2f}%")

# Create directories for saving outputs
def create_directories():
    """Create necessary directories for saving outputs"""
    directories = [
        'model_outputs',
        'model_outputs/plots',
        'model_outputs/reports',
        'model_outputs/model'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("\n✓ Created directory structure for saving outputs")

create_directories()

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name="model"):
    '''The function will take model, x train, x test, y train, y test
    and then it will fit the model, then make predictions on the trained model,
    it will then print roc-auc score of train and test, then plot the roc, auc curve,
    print confusion matrix for train and test, then print classification report for train and test,
    and finally it will return the following scores as a list:
    precision_train, precision_test, recall_train, recall_test, acc_train, acc_test, roc_auc_train, roc_auc_test, F1_train, F1_test
    '''
    
    print("\n" + "=" * 70)
    print(f"EVALUATING MODEL: {model_name.upper()}")
    print("=" * 70)
    
    # fit the model on the training data
    print("Training the model...")
    model.fit(X_train, y_train)
    print("✓ Model training completed")

    # make predictions on the test data
    print("\nMaking predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    pred_prob_train = model.predict_proba(X_train)[:, 1]
    pred_prob_test = model.predict_proba(X_test)[:, 1]
    print("✓ Predictions completed")

    # calculate ROC AUC score
    roc_auc_train = roc_auc_score(y_train, y_pred_train)
    roc_auc_test = roc_auc_score(y_test, y_pred_test)
    print("\n" + "-" * 50)
    print("ROC AUC SCORES")
    print("-" * 50)
    print(f"Train ROC AUC: {roc_auc_train:.4f}")
    print(f"Test ROC AUC:  {roc_auc_test:.4f}")

    # plot the ROC curve
    print("\nGenerating ROC Curve...")
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, pred_prob_train)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_prob_test)
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Classifier')
    plt.plot(fpr_train, tpr_train, label=f"Train (AUC = {roc_auc_train:.3f})", linewidth=2, color='blue')
    plt.plot(fpr_test, tpr_test, label=f"Test (AUC = {roc_auc_test:.3f})", linewidth=2, color='red', linestyle='--')
    plt.legend(loc='lower right', fontsize=12)
    plt.title(f"ROC Curve - {model_name.replace('_', ' ').title()}", fontsize=16, pad=20)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save ROC curve plot
    roc_curve_path = f'model_outputs/plots/{model_name}_roc_curve.png'
    plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ ROC curve saved to: {roc_curve_path}")
    plt.show()

    # calculate confusion matrix
    print("\nGenerating Confusion Matrices...")
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    print("\n" + "-" * 50)
    print("CONFUSION MATRICES")
    print("-" * 50)
    
    # Train confusion matrix
    sns.heatmap(cm_train, annot=True, fmt='d', xticklabels=['Ham (0)', 'Spam (1)'], 
                yticklabels=['Ham (0)', 'Spam (1)'], cmap="Blues", ax=ax[0], 
                cbar_kws={'label': 'Count'}, annot_kws={"size": 14})
    ax[0].set_xlabel("Predicted Label", fontsize=12)
    ax[0].set_ylabel("True Label", fontsize=12)
    ax[0].set_title("Train Confusion Matrix", fontsize=14, pad=15)
    
    # Test confusion matrix
    sns.heatmap(cm_test, annot=True, fmt='d', xticklabels=['Ham (0)', 'Spam (1)'], 
                yticklabels=['Ham (0)', 'Spam (1)'], cmap="Greens", ax=ax[1],
                cbar_kws={'label': 'Count'}, annot_kws={"size": 14})
    ax[1].set_xlabel("Predicted Label", fontsize=12)
    ax[1].set_ylabel("True Label", fontsize=12)
    ax[1].set_title("Test Confusion Matrix", fontsize=14, pad=15)

    plt.tight_layout()
    
    # Save confusion matrix plot
    cm_path = f'model_outputs/plots/{model_name}_confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Confusion matrix saved to: {cm_path}")
    plt.show()

    # Save confusion matrix as CSV
    cm_train_df = pd.DataFrame(cm_train, 
                               index=['Actual Ham', 'Actual Spam'],
                               columns=['Predicted Ham', 'Predicted Spam'])
    cm_test_df = pd.DataFrame(cm_test,
                              index=['Actual Ham', 'Actual Spam'],
                              columns=['Predicted Ham', 'Predicted Spam'])
    
    cm_train_path = f'model_outputs/reports/{model_name}_confusion_matrix_train.csv'
    cm_test_path = f'model_outputs/reports/{model_name}_confusion_matrix_test.csv'
    
    cm_train_df.to_csv(cm_train_path)
    cm_test_df.to_csv(cm_test_path)
    print(f"✓ Confusion matrix data saved to CSV files")

    # calculate classification report
    print("\nGenerating Classification Reports...")
    cr_train = classification_report(y_train, y_pred_train, output_dict=True)
    cr_test = classification_report(y_test, y_pred_test, output_dict=True)
    
    print("\n" + "-" * 50)
    print("TRAIN CLASSIFICATION REPORT")
    print("-" * 50)
    crt = pd.DataFrame(cr_train).T
    print(crt.to_markdown())
    
    print("\n" + "-" * 50)
    print("TEST CLASSIFICATION REPORT")
    print("-" * 50)
    crt2 = pd.DataFrame(cr_test).T
    print(crt2.to_markdown())
    
    # Save classification reports as CSV and Excel
    cr_train_df = pd.DataFrame(cr_train).T
    cr_test_df = pd.DataFrame(cr_test).T
    
    cr_train_path_csv = f'model_outputs/reports/{model_name}_classification_report_train.csv'
    cr_test_path_csv = f'model_outputs/reports/{model_name}_classification_report_test.csv'
    cr_excel_path = f'model_outputs/reports/{model_name}_classification_reports.xlsx'
    
    cr_train_df.to_csv(cr_train_path_csv)
    cr_test_df.to_csv(cr_test_path_csv)
    
    # Also save as Excel for better formatting
    with pd.ExcelWriter(cr_excel_path, engine='openpyxl') as writer:
        cr_train_df.to_excel(writer, sheet_name='Train_Report')
        cr_test_df.to_excel(writer, sheet_name='Test_Report')
    
    print(f"\n✓ Classification reports saved to CSV and Excel files")

    # Calculate all metrics
    precision_train = cr_train['weighted avg']['precision']
    precision_test = cr_test['weighted avg']['precision']
    recall_train = cr_train['weighted avg']['recall']
    recall_test = cr_test['weighted avg']['recall']
    acc_train = accuracy_score(y_true=y_train, y_pred=y_pred_train)
    acc_test = accuracy_score(y_true=y_test, y_pred=y_pred_test)
    F1_train = cr_train['weighted avg']['f1-score']
    F1_test = cr_test['weighted avg']['f1-score']

    # Save all metrics in a summary file
    metrics_summary = {
        'model_name': model_name,
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_spam_percentage': f"{(y_train.sum()/len(y_train))*100:.2f}%",
        'test_spam_percentage': f"{(y_test.sum()/len(y_test))*100:.2f}%",
        'metrics': {
            'train_accuracy': float(acc_train),
            'test_accuracy': float(acc_test),
            'train_precision': float(precision_train),
            'test_precision': float(precision_test),
            'train_recall': float(recall_train),
            'test_recall': float(recall_test),
            'train_f1_score': float(F1_train),
            'test_f1_score': float(F1_test),
            'train_roc_auc': float(roc_auc_train),
            'test_roc_auc': float(roc_auc_test)
        },
        'confusion_matrix': {
            'train': {
                'true_negative': int(cm_train[0, 0]),
                'false_positive': int(cm_train[0, 1]),
                'false_negative': int(cm_train[1, 0]),
                'true_positive': int(cm_train[1, 1])
            },
            'test': {
                'true_negative': int(cm_test[0, 0]),
                'false_positive': int(cm_test[0, 1]),
                'false_negative': int(cm_test[1, 0]),
                'true_positive': int(cm_test[1, 1])
            }
        }
    }
    
    # Save metrics summary as JSON
    metrics_json_path = f'model_outputs/reports/{model_name}_metrics_summary.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    # Save metrics summary as CSV
    metrics_df = pd.DataFrame([{
        'Model': model_name,
        'Train_Accuracy': acc_train,
        'Test_Accuracy': acc_test,
        'Train_Precision': precision_train,
        'Test_Precision': precision_test,
        'Train_Recall': recall_train,
        'Test_Recall': recall_test,
        'Train_F1_Score': F1_train,
        'Test_F1_Score': F1_test,
        'Train_ROC_AUC': roc_auc_train,
        'Test_ROC_AUC': roc_auc_test
    }])
    
    metrics_csv_path = f'model_outputs/reports/{model_name}_metrics_summary.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    
    print(f"✓ Metrics summary saved to JSON and CSV files")

    # Save the trained model
    model_path = f'model_outputs/model/{model_name}.pkl'
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved successfully at: {model_path}")

    # Create a summary DataFrame for display
    summary_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        'Train': [acc_train, precision_train, recall_train, F1_train, roc_auc_train],
        'Test': [acc_test, precision_test, recall_test, F1_test, roc_auc_test]
    })
    
    summary_df['Train'] = summary_df['Train'].apply(lambda x: f"{x:.4f}")
    summary_df['Test'] = summary_df['Test'].apply(lambda x: f"{x:.4f}")
    
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE SUMMARY")
    print("=" * 70)
    print(summary_df.to_markdown(index=False))
    
    model_score = [precision_train, precision_test, recall_train, recall_test, 
                   acc_train, acc_test, roc_auc_train, roc_auc_test, F1_train, F1_test]
    
    return model_score

# ML Model - 1 Implementation
print("\n" + "=" * 70)
print("CREATING AND TRAINING MULTINOMIAL NAIVE BAYES PIPELINE")
print("=" * 70)
# Create a machine learning pipeline using scikit-learn, combining text vectorization (CountVectorizer)
# and a Multinomial Naive Bayes classifier for email spam detection.
clf = Pipeline([
    ('vectorizer', CountVectorizer()),  # Step 1: Text data transformation
    ('nb', MultinomialNB())  # Step 2: Classification using Naive Bayes
])

print("Pipeline created with:")
print("1. CountVectorizer - for text vectorization")
print("2. MultinomialNB - Naive Bayes classifier")

# Model is trained (fit) and predicted in the evaluate model
MultinomialNB_score = evaluate_model(clf, X_train, X_test, y_train, y_test, model_name="multinomial_nb_spam_detector")

# Function to load and use the saved model
def load_and_predict(model_path, new_messages):
    """
    Load a saved model and make predictions on new messages
    
    Parameters:
    model_path: Path to the saved model
    new_messages: List of new messages to classify
    
    Returns:
    Predictions and probabilities
    """
    print("\n" + "=" * 70)
    print("LOADING SAVED MODEL AND MAKING PREDICTIONS")
    print("=" * 70)
    
    # Load the model
    try:
        loaded_model = joblib.load(model_path)
        print(f"✓ Model loaded successfully from: {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None
    
    # Make predictions
    predictions = loaded_model.predict(new_messages)
    probabilities = loaded_model.predict_proba(new_messages)
    
    # Create a DataFrame with results
    results_df = pd.DataFrame({
        'Message': new_messages,
        'Prediction': ['Spam' if pred == 1 else 'Ham' for pred in predictions],
        'Spam_Probability': probabilities[:, 1],
        'Ham_Probability': probabilities[:, 0],
        'Confidence': np.max(probabilities, axis=1)
    })
    
    # Sort by Spam probability (descending)
    results_df = results_df.sort_values('Spam_Probability', ascending=False).reset_index(drop=True)
    
    return results_df

# Function to save predictions to file
def save_predictions(predictions_df, filename):
    """Save predictions to CSV and Excel files"""
    # Save to CSV
    csv_path = f'model_outputs/reports/{filename}.csv'
    predictions_df.to_csv(csv_path, index=False)
    
    # Save to Excel with formatting
    excel_path = f'model_outputs/reports/{filename}.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        predictions_df.to_excel(writer, sheet_name='Predictions', index=False)
        
        # Get the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['Predictions']
        
        # Add some formatting
        for column in predictions_df.columns:
            column_width = max(predictions_df[column].astype(str).map(len).max(), len(column)) + 2
            col_idx = predictions_df.columns.get_loc(column)
            worksheet.column_dimensions[chr(65 + col_idx)].width = column_width
    
    print(f"✓ Predictions saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  Excel: {excel_path}")
    
    return csv_path, excel_path

# Example of using the saved model
print("\n" + "=" * 70)
print("MODEL SAVING AND LOADING DEMONSTRATION")
print("=" * 70)

# Example messages for prediction
example_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim your prize!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your bank account needs verification. Please click the link to update your information.",
    "Can you send me the report by EOD today?",
    "FREE entry to our weekly prize draw! Text WIN to 88888 now!",
    "Hi John, I'll be 5 minutes late for our meeting. Sorry!",
    "You have been selected for a special offer! Buy one get one free on all products!",
    "Meeting rescheduled to 3 PM tomorrow. Please confirm your availability.",
    "Claim your $1000 Walmart gift card! Limited time offer!",
    "The project deadline has been extended to next Friday."
]

# Load the saved model and make predictions
try:
    model_path = 'model_outputs/model/multinomial_nb_spam_detector.pkl'
    results = load_and_predict(model_path, example_messages)
    
    if results is not None:
        print("\n" + "-" * 70)
        print("PREDICTION RESULTS ON EXAMPLE MESSAGES")
        print("-" * 70)
        print(results.to_string(index=False))
        
        # Display summary
        print("\n" + "-" * 70)
        print("PREDICTION SUMMARY")
        print("-" * 70)
        spam_count = (results['Prediction'] == 'Spam').sum()
        ham_count = (results['Prediction'] == 'Ham').sum()
        print(f"Spam messages: {spam_count}")
        print(f"Ham messages: {ham_count}")
        print(f"Spam percentage: {(spam_count/len(results))*100:.1f}%")
        
        # Save predictions
        csv_path, excel_path = save_predictions(results, 'example_predictions')
        
except Exception as e:
    print(f"Error in prediction demonstration: {e}")

# Generate Word Cloud for spam and ham messages
print("\n" + "=" * 70)
print("GENERATING WORD CLOUDS")
print("=" * 70)

def generate_word_cloud(texts, title, filename):
    """Generate and save word cloud"""
    # Combine all texts
    all_text = ' '.join(texts)
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white', 
        stopwords=STOPWORDS, 
        max_words=100,
        colormap='viridis'
    ).generate(all_text)
    
    # Plot word cloud
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud - {title}', fontsize=16, pad=20)
    plt.axis('off')
    
    # Save word cloud
    wordcloud_path = f'model_outputs/plots/{filename}.png'
    plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ {title} word cloud saved to: {wordcloud_path}")
    plt.show()

# Separate spam and ham messages
spam_messages = df[df['Spam'] == 1]['Message'].tolist()
ham_messages = df[df['Spam'] == 0]['Message'].tolist()

print(f"\nGenerating word clouds:")
print(f"Spam messages: {len(spam_messages)}")
print(f"Ham messages: {len(ham_messages)}")

# Generate spam word cloud
generate_word_cloud(spam_messages, 'Spam Messages', 'spam_word_cloud')

# Generate ham word cloud
generate_word_cloud(ham_messages, 'Ham Messages', 'ham_word_cloud')

# Final summary
print("\n" + "=" * 70)
print("PROCESS COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nFILES GENERATED:")
print("-" * 70)

# List generated files
for root, dirs, files in os.walk('model_outputs'):
    level = root.replace('model_outputs', '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        if file.endswith('.png'):
            print(f"{subindent}📊 {file}")
        elif file.endswith('.pkl'):
            print(f"{subindent}🤖 {file}")
        elif file.endswith('.csv') or file.endswith('.json'):
            print(f"{subindent}📄 {file}")
        elif file.endswith('.xlsx'):
            print(f"{subindent}📊 {file}")

print("\n" + "=" * 70)
print("NEXT STEPS:")
print("=" * 70)
print("1. Use the saved model for new predictions:")
print("   model = joblib.load('model_outputs/model/multinomial_nb_spam_detector.pkl')")
print("   predictions = model.predict(new_messages)")
print("\n2. Check the reports folder for detailed performance metrics")
print("\n3. View the plots folder for visualizations")
print("\n" + "=" * 70)