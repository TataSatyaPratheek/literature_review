"""Metrics for evaluating model performance."""
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def calculate_accuracy(predictions, targets):
    """
    Calculate classification accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        
    Returns:
        Accuracy as percentage
    """
    _, predicted = predictions.max(1)
    correct = predicted.eq(targets).sum().item()
    total = targets.size(0)
    
    return 100.0 * correct / total

def calculate_top_k_accuracy(predictions, targets, k=5):
    """
    Calculate top-k accuracy.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy as percentage
    """
    _, top_k_pred = predictions.topk(k, dim=1)
    targets_expanded = targets.view(-1, 1).expand_as(top_k_pred)
    correct = top_k_pred.eq(targets_expanded).sum().item()
    total = targets.size(0)
    
    return 100.0 * correct / total

def calculate_precision_recall_f1(predictions, targets, num_classes):
    """
    Calculate precision, recall, and F1 score per class.
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with metrics per class
    """
    _, predicted = predictions.max(1)
    
    # Convert to numpy
    pred_np = predicted.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Calculate metrics per class
    metrics = {}
    for c in range(num_classes):
        tp = ((pred_np == c) & (targets_np == c)).sum()
        fp = ((pred_np == c) & (targets_np != c)).sum()
        fn = ((pred_np != c) & (targets_np == c)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'class_{c}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics

def calculate_confusion_matrix(predictions, targets, num_classes):
    """
    Calculate confusion matrix.
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Confusion matrix as numpy array
    """
    _, predicted = predictions.max(1)
    
    pred_np = predicted.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    return confusion_matrix(targets_np, pred_np, labels=range(num_classes))

def calculate_metrics(predictions, targets, num_classes):
    """
    Calculate all metrics at once.
    
    Args:
        predictions: Model predictions (logits)
        targets: Ground truth labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'accuracy': calculate_accuracy(predictions, targets),
        'top_5_accuracy': calculate_top_k_accuracy(predictions, targets, k=5),
        'confusion_matrix': calculate_confusion_matrix(predictions, targets, num_classes),
        'per_class': calculate_precision_recall_f1(predictions, targets, num_classes)
    }
    
    return metrics
