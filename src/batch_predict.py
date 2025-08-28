"""
Batch Prediction Module for Requirements Evaluation

This module allows batch processing of requirements through the trained model
to generate predictions in bulk, with options for output formats.
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from test_req_model import load_model, predict_with_confidence
from tqdm import tqdm

def batch_predict(input_file, output_file=None, format="excel", confidence_threshold=0.0):
    """
    Run batch prediction on requirements from various input formats
    
    Parameters:
        input_file (str): Path to the input file (JSONL, CSV, or Excel)
        output_file (str): Path to save the results (optional)
        format (str): Output format - "excel", "csv", or "jsonl"
        confidence_threshold (float): Minimum confidence to include in results
        
    Returns:
        dict: Prediction statistics
    """
    print(f"\nRunning batch prediction on {input_file}")
    
    # Load requirements from file
    requirements = []
    
    # Parse based on file type
    if input_file.lower().endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'text' in data:
                        # Copy all fields from the original record
                        requirement = data.copy()
                        requirements.append(requirement)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line: {line}")
                    
    elif input_file.lower().endswith('.csv'):
        df = pd.read_csv(input_file)
        for _, row in df.iterrows():
            req = {}
            for col in df.columns:
                req[col] = row[col]
            if 'text' in req or 'Primary Text' in req:
                # Use 'Primary Text' as fallback if 'text' not present
                if 'text' not in req and 'Primary Text' in req:
                    req['text'] = req['Primary Text']
                requirements.append(req)
    
    elif input_file.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
        for _, row in df.iterrows():
            req = {}
            for col in df.columns:
                req[col] = row[col]
            if 'text' in req or 'Primary Text' in req:
                # Use 'Primary Text' as fallback if 'text' not present
                if 'text' not in req and 'Primary Text' in req:
                    req['text'] = req['Primary Text']
                requirements.append(req)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    print(f"Loaded {len(requirements)} requirements for processing")
    
    # Process each requirement
    results = []
    stats = {
        "total": len(requirements),
        "agreed": 0,
        "partly_agreed": 0,
        "not_agreed": 0,
        "low_confidence": 0
    }
    
    # Use tqdm for progress bar
    for req in tqdm(requirements, desc="Processing requirements"):
        # Get prediction
        text = req.get('text', '')
        status, confidences = predict_with_confidence(text)
        
        # Get confidence scores
        confidence = max(confidences.values())
        
        # Check if confidence is above threshold
        if confidence >= confidence_threshold:
            # Count by status
            if status == "agreed":
                stats["agreed"] += 1
            elif status == "partly agreed":
                stats["partly_agreed"] += 1
            elif status == "not agreed":
                stats["not_agreed"] += 1
        else:
            stats["low_confidence"] += 1
        
        # Add prediction to requirement
        result = req.copy()
        result['predicted_status'] = status
        result['prediction_confidence'] = confidence
        
        # Add individual confidences
        for status_key, conf_value in confidences.items():
            result[f'confidence_{status_key.replace(" ", "_")}'] = conf_value
            
        results.append(result)
    
    # Save results if output file provided
    if output_file:
        if format.lower() == "excel":
            # Save as Excel
            df = pd.DataFrame(results)
            df.to_excel(output_file, index=False)
            print(f"Results saved to Excel: {output_file}")
            
        elif format.lower() == "csv":
            # Save as CSV
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"Results saved to CSV: {output_file}")
            
        elif format.lower() == "jsonl":
            # Save as JSONL
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"Results saved to JSONL: {output_file}")
            
        else:
            print(f"Warning: Unsupported output format '{format}'. Results not saved.")
    
    # Print statistics
    print("\nPrediction Results:")
    print(f"Total requirements: {stats['total']}")
    print(f"Agreed: {stats['agreed']} ({100 * stats['agreed'] / stats['total']:.1f}%)")
    print(f"Partly Agreed: {stats['partly_agreed']} ({100 * stats['partly_agreed'] / stats['total']:.1f}%)")
    print(f"Not Agreed: {stats['not_agreed']} ({100 * stats['not_agreed'] / stats['total']:.1f}%)")
    if stats['low_confidence'] > 0:
        print(f"Low confidence: {stats['low_confidence']} ({100 * stats['low_confidence'] / stats['total']:.1f}%)")
    
    return stats, results

def create_confusion_matrix(results, actual_field='supplier_status', predicted_field='predicted_status'):
    """
    Create a confusion matrix from prediction results
    
    Parameters:
        results (list): List of prediction results
        actual_field (str): Field name containing the actual status
        predicted_field (str): Field name containing the predicted status
        
    Returns:
        tuple: (confusion_matrix, categories)
    """
    # Filter results that have both actual and predicted values
    valid_results = [r for r in results if actual_field in r and r[actual_field] and 
                    predicted_field in r and r[predicted_field]]
    
    if not valid_results:
        return None, None
    
    # Get unique categories (sorted for consistent matrix)
    categories = sorted(set([r[actual_field].lower() for r in valid_results] + 
                          [r[predicted_field].lower() for r in valid_results]))
    
    # Initialize confusion matrix
    matrix = np.zeros((len(categories), len(categories)))
    
    # Fill confusion matrix
    for result in valid_results:
        actual = result[actual_field].lower()
        predicted = result[predicted_field].lower()
        
        # Skip if either value is not in categories
        if actual not in categories or predicted not in categories:
            continue
            
        actual_idx = categories.index(actual)
        predicted_idx = categories.index(predicted)
        matrix[actual_idx, predicted_idx] += 1
    
    return matrix, categories

def calculate_metrics(confusion_matrix, categories):
    """
    Calculate accuracy, precision, recall and F1 score from confusion matrix
    
    Parameters:
        confusion_matrix: The confusion matrix
        categories: List of category names
        
    Returns:
        dict: Metrics
    """
    if confusion_matrix is None:
        return None
        
    # Overall accuracy
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    
    metrics = {
        'accuracy': accuracy,
        'class_metrics': {}
    }
    
    # Per-class metrics
    for i, category in enumerate(categories):
        # True positives, false positives, false negatives
        tp = confusion_matrix[i, i]
        fp = np.sum(confusion_matrix[:, i]) - tp
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # Calculate metrics with handling for division by zero
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['class_metrics'][category] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics

def generate_comparison_report(input_file, output_file=None):
    """
    Generate a report comparing model predictions with actual values
    
    Parameters:
        input_file (str): Path to the input file (already contains predictions)
        output_file (str): Path to save the report (optional)
        
    Returns:
        dict: Report metrics
    """
    # Load results
    if input_file.lower().endswith('.jsonl'):
        with open(input_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
    elif input_file.lower().endswith('.csv'):
        df = pd.read_csv(input_file)
        results = df.to_dict('records')
    elif input_file.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(input_file)
        results = df.to_dict('records')
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    # Create confusion matrix
    confusion_matrix, categories = create_confusion_matrix(results)
    
    # Calculate metrics
    metrics = calculate_metrics(confusion_matrix, categories)
    
    # Generate report
    if metrics:
        print("\nModel Evaluation Report:")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print("\nPer-class Metrics:")
        for category, class_metrics in metrics['class_metrics'].items():
            print(f"  {category}:")
            print(f"    Precision: {class_metrics['precision']:.4f}")
            print(f"    Recall: {class_metrics['recall']:.4f}")
            print(f"    F1 Score: {class_metrics['f1']:.4f}")
            
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write("Model Evaluation Report\n")
                f.write("-----------------------\n\n")
                f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n\n")
                f.write("Per-class Metrics:\n")
                for category, class_metrics in metrics['class_metrics'].items():
                    f.write(f"  {category}:\n")
                    f.write(f"    Precision: {class_metrics['precision']:.4f}\n")
                    f.write(f"    Recall: {class_metrics['recall']:.4f}\n")
                    f.write(f"    F1 Score: {class_metrics['f1']:.4f}\n\n")
                    
                # Add confusion matrix
                f.write("Confusion Matrix:\n")
                f.write("  Actual (rows) vs Predicted (columns)\n\n")
                f.write("  " + "\t".join(categories) + "\n")
                for i, category in enumerate(categories):
                    f.write(f"{category}\t" + "\t".join([str(int(confusion_matrix[i, j])) for j in range(len(categories))]) + "\n")
    
    return metrics if metrics else None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch prediction for requirements evaluation")
    parser.add_argument('input_file', help='Path to input file (JSONL, CSV, or Excel)')
    parser.add_argument('--output', '-o', help='Path to output file')
    parser.add_argument('--format', '-f', default='excel', choices=['excel', 'csv', 'jsonl'], 
                      help='Output format (default: excel)')
    parser.add_argument('--threshold', '-t', type=float, default=0.0, 
                      help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--report', '-r', action='store_true',
                      help='Generate evaluation report (if input has actual statuses)')
    
    args = parser.parse_args()
    
    # Run batch prediction
    stats, results = batch_predict(args.input_file, args.output, args.format, args.threshold)
    
    # Generate report if requested
    if args.report and args.output:
        report_file = os.path.splitext(args.output)[0] + "_report.txt"
        generate_comparison_report(args.output, report_file)
