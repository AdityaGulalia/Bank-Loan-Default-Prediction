from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def check_top_risk(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    results = pd.DataFrame({'actual': y_test, 'prob': probs})
    
    # Sort by highest risk
    top_10_percent = results.sort_values(by='prob', ascending=False).head(len(results)//10)
    
    capture_rate = top_10_percent['actual'].sum() / y_test.sum()
    print(f"Top 10% Risk Group: Captures {capture_rate:.2%} of all actual defaults.")

def perform_evaluation(model, X_test, y_test, threshold=0.3):
    """
    Interactive evaluation with adjustable threshold.
    """

    # ----------------------------
    # Predictions with threshold
    # ----------------------------
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # ----------------------------
    # Metrics
    # ----------------------------
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)

    # ----------------------------
    # Print Results
    # ----------------------------
    print("\n" + "=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"Threshold : {threshold}")
    print(f"ACCURACY  : {accuracy:.4f}")
    print(f"PRECISION : {precision:.4f}")
    print(f"RECALL    : {recall:.4f}")
    print(f"F1 SCORE  : {f1:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")
    
    probs_class1 = probabilities[y_test == 1]
    probs_class0 = probabilities[y_test == 0]

    print("Mean prob for actual class 1:", np.mean(probs_class1))
    print("Mean prob for actual class 0:", np.mean(probs_class0))  
    
    for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
        preds = (probabilities >= t).astype(int)
        rec = recall_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        print(f"Threshold {t}: Precision={prec:.2f}, Recall={rec:.2f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    # ----------------------------
    # Confusion Matrix Plot
    # ----------------------------
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show(block=True)

    # ----------------------------
    # ROC Curve Plot
    # ----------------------------
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show(block=True)

    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }