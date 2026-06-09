import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from ultralytics import YOLO

# To prevent plots from blocking
import matplotlib
matplotlib.use('Agg')

def get_true_labels_and_predictions(model, test_dir):
    true_labels = []
    pred_probs = []
    pred_labels = []
    
    classes = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    for cls_name in classes:
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
            
        files = glob.glob(os.path.join(cls_dir, '*.*'))
        for f in files:
            # Predict
            results = model.predict(f, verbose=False)
            probs = results[0].probs.data.cpu().numpy()
            pred_class = np.argmax(probs)
            
            true_labels.append(class_to_idx[cls_name])
            pred_probs.append(probs)
            pred_labels.append(pred_class)
            
    return np.array(true_labels), np.array(pred_labels), np.array(pred_probs), classes

def evaluate_and_plot(true_labels, pred_labels, pred_probs, classes, model_name, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    f1 = f1_score(true_labels, pred_labels, average='macro')
    
    # Save metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    
    # Confusion Matrix (Absolute)
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(os.path.join(results_dir, f'cm_{model_name}.png'))
    plt.close()

    # Confusion Matrix (Normalized)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8,6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Matriz de Confusión Normalizada - {model_name}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.savefig(os.path.join(results_dir, f'cm_norm_{model_name}.png'))
    plt.close()
    
    # ROC Curve & AUC
    if len(classes) == 2:
        try:
            pos_idx = classes.index('PNEUMONIA')
        except ValueError:
            pos_idx = 1
            
        fpr, tpr, _ = roc_curve(true_labels, pred_probs[:, pos_idx], pos_label=pos_idx)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title(f'Curva ROC - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(results_dir, f'roc_{model_name}.png'))
        plt.close()
        
        metrics['AUC'] = roc_auc
        
    return metrics

def main():
    epochs = 5 # Using 5 for time constraints, can be increased.
    img_size = 224 # Default yolo cls size
    
    # 1. Train on Original Dataset
    print("Training/Loading Original Dataset...")
    model_orig = YOLO('yolov8n-cls.pt')
    orig_data_path = os.path.abspath('Chest-X-Rays-4')
    orig_model_path = os.path.join('runs_classification', 'original_model', 'weights', 'best.pt')
    if not os.path.exists(orig_model_path):
        model_orig.train(data=orig_data_path, epochs=epochs, imgsz=img_size, project='runs_classification', name='original_model')
    else:
        print("Modelo original ya entrenado. Cargando pesos...")
        model_orig = YOLO(orig_model_path)
    
    # 2. Train on Preprocessed Dataset
    print("Training/Loading Preprocessed Dataset...")
    model_prep = YOLO('yolov8n-cls.pt')
    prep_data_path = os.path.abspath('Chest-X-Rays-4-Preprocessed')
    prep_model_path = os.path.join('runs_classification', 'preprocessed_model', 'weights', 'best.pt')
    if not os.path.exists(prep_model_path):
        model_prep.train(data=prep_data_path, epochs=epochs, imgsz=img_size, project='runs_classification', name='preprocessed_model')
    else:
        print("Modelo preprocesado ya entrenado. Cargando pesos...")
        model_prep = YOLO(prep_model_path)
    
    # 3. Evaluate Original
    print("Evaluating Original Model...")
    true_orig, pred_orig, probs_orig, classes = get_true_labels_and_predictions(model_orig, os.path.join(orig_data_path, 'test'))
    metrics_orig = evaluate_and_plot(true_orig, pred_orig, probs_orig, classes, 'Original', 'results')
    
    # 4. Evaluate Preprocessed
    print("Evaluating Preprocessed Model...")
    # Get the best model path
    prep_model_path = os.path.join('runs_classification', 'preprocessed_model', 'weights', 'best.pt')
    if os.path.exists(prep_model_path):
        model_prep = YOLO(prep_model_path)
    true_prep, pred_prep, probs_prep, _ = get_true_labels_and_predictions(model_prep, os.path.join(prep_data_path, 'test'))
    metrics_prep = evaluate_and_plot(true_prep, pred_prep, probs_prep, classes, 'Preprocesado', 'results')
    
    # 5. Cross Evaluate: Original Model on Preprocessed Data
    print("Evaluating Original Model on Preprocessed Data...")
    orig_model_path = os.path.join('runs_classification', 'original_model', 'weights', 'best.pt')
    if os.path.exists(orig_model_path):
        model_orig = YOLO(orig_model_path)
    true_orig_prep, pred_orig_prep, probs_orig_prep, _ = get_true_labels_and_predictions(model_orig, os.path.join(prep_data_path, 'test'))
    metrics_orig_prep = evaluate_and_plot(true_orig_prep, pred_orig_prep, probs_orig_prep, classes, 'Original_sobre_Preprocesado', 'results')
    
    # 6. Cross Evaluate: Preprocessed Model on Original Data
    print("Evaluating Preprocessed Model on Original Data...")
    if os.path.exists(prep_model_path):
        model_prep = YOLO(prep_model_path)
    true_prep_orig, pred_prep_orig, probs_prep_orig, _ = get_true_labels_and_predictions(model_prep, os.path.join(orig_data_path, 'test'))
    metrics_prep_orig = evaluate_and_plot(true_prep_orig, pred_prep_orig, probs_prep_orig, classes, 'Preprocesado_sobre_Original', 'results')

    # 7. Save combined metrics to CSV
    df = pd.DataFrame([metrics_orig, metrics_prep, metrics_orig_prep, metrics_prep_orig])
    df.to_csv('results/metrics.csv', index=False)
    print("Metrics saved to results/metrics.csv")
    print(df)

if __name__ == '__main__':
    main()
