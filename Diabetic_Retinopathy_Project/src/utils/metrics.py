# ============================================================================
# /src/utils/metrics.py
# Evaluation Metrics for DR Classification
# ============================================================================

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, cohen_kappa_score, classification_report
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DRMetrics:
    """Comprehensive metrics for Diabetic Retinopathy classification"""
    
    def __init__(self, num_classes: int = 5, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
    
    def calculate_all_metrics(
        self,
        y_true: Union[torch.Tensor, np.ndarray],
        y_pred: Union[torch.Tensor, np.ndarray],
        y_probs: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, Union[float, np.ndarray, Dict]]:
        """Calculate comprehensive metrics for DR classification"""
        
        # Convert to numpy if needed
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.cpu().numpy()
        if y_probs is not None and isinstance(y_probs, torch.Tensor):
            y_probs = y_probs.cpu().numpy()
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['per_class'] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support
        }
        
        # Averaged metrics
        metrics['macro_precision'] = np.mean(precision)
        metrics['macro_recall'] = np.mean(recall)
        metrics['macro_f1'] = np.mean(f1)
        
        # Weighted metrics
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['weighted_precision'] = precision_w
        metrics['weighted_recall'] = recall_w
        metrics['weighted_f1'] = f1_w
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Cohen's Kappa (important for ordinal classification)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # Quadratic weighted kappa (for ordinal data)
        metrics['quadratic_kappa'] = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        # AUC metrics (if probabilities provided)
        if y_probs is not None:
            auc_metrics = self._calculate_auc_metrics(y_true, y_probs)
            metrics.update(auc_metrics)
        
        # DR-specific metrics
        dr_metrics = self._calculate_dr_specific_metrics(y_true, y_pred)
        metrics.update(dr_metrics)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        return metrics
    
    def _calculate_auc_metrics(
        self, 
        y_true: np.ndarray, 
        y_probs: np.ndarray
    ) -> Dict[str, float]:
        """Calculate AUC metrics"""
        auc_metrics = {}
        
        try:
            # Multi-class AUC (one-vs-rest)
            y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
            
            # Macro AUC
            auc_scores = []
            for i in range(self.num_classes):
                if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class is present
                    auc = roc_auc_score(y_true_bin[:, i], y_probs[:, i])
                    auc_scores.append(auc)
            
            if auc_scores:
                auc_metrics['macro_auc'] = np.mean(auc_scores)
                auc_metrics['per_class_auc'] = auc_scores
            
            # Multi-class AUC
            if self.num_classes == 2:
                auc_metrics['auc'] = roc_auc_score(y_true, y_probs[:, 1])
            else:
                auc_metrics['auc_ovr'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='macro'
                )
                auc_metrics['auc_ovo'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovo', average='macro'
                )
        
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
        
        return auc_metrics
    
    def _calculate_dr_specific_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate DR-specific metrics"""
        dr_metrics = {}
        
        # Binary classification: No DR vs Any DR
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = (y_pred > 0).astype(int)
        
        dr_metrics['binary_accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        dr_metrics['binary_precision'] = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary'
        )[0]
        dr_metrics['binary_recall'] = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary'
        )[1]
        dr_metrics['binary_f1'] = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average='binary'
        )[2]
        
        # Referrable DR detection (moderate or worse)
        y_true_referrable = (y_true >= 2).astype(int)  # Moderate, Severe, PDR
        y_pred_referrable = (y_pred >= 2).astype(int)
        
        dr_metrics['referrable_accuracy'] = accuracy_score(y_true_referrable, y_pred_referrable)
        dr_metrics['referrable_precision'] = precision_recall_fscore_support(
            y_true_referrable, y_pred_referrable, average='binary'
        )[0]
        dr_metrics['referrable_recall'] = precision_recall_fscore_support(
            y_true_referrable, y_pred_referrable, average='binary'
        )[1]
        dr_metrics['referrable_f1'] = precision_recall_fscore_support(
            y_true_referrable, y_pred_referrable, average='binary'
        )[2]
        
        # Mean Absolute Error (for ordinal nature)
        dr_metrics['mae'] = np.mean(np.abs(y_true - y_pred))
        
        # Off-by-one accuracy (acceptable for clinical use)
        off_by_one = np.abs(y_true - y_pred) <= 1
        dr_metrics['off_by_one_accuracy'] = np.mean(off_by_one)
        
        return dr_metrics
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        normalize: bool = True,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot classification report as heatmap"""
        
        report = classification_report(
            y_true, y_pred, target_names=self.class_names, output_dict=True
        )
        
        # Convert to DataFrame for plotting
        df = pd.DataFrame(report).iloc[:-1, :-3].T  # Remove accuracy, macro avg, weighted avg
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            df,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            ax=ax
        )
        
        ax.set_title('Classification Report')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Classes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_metrics_summary(
        self, 
        metrics: Dict[str, Union[float, np.ndarray, Dict]]
    ) -> pd.DataFrame:
        """Create a summary DataFrame of all metrics"""
        
        summary_data = []
        
        # Overall metrics
        overall_metrics = [
            'accuracy', 'macro_precision', 'macro_recall', 'macro_f1',
            'weighted_precision', 'weighted_recall', 'weighted_f1',
            'kappa', 'quadratic_kappa', 'mae', 'off_by_one_accuracy'
        ]
        
        for metric in overall_metrics:
            if metric in metrics:
                summary_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': f"{metrics[metric]:.4f}"
                })
        
        # DR-specific metrics
        dr_specific = [
            'binary_accuracy', 'binary_precision', 'binary_recall', 'binary_f1',
            'referrable_accuracy', 'referrable_precision', 'referrable_recall', 'referrable_f1'
        ]
        
        for metric in dr_specific:
            if metric in metrics:
                summary_data.append({
                    'Metric': metric.replace('_', ' ').title(),
                    'Value': f"{metrics[metric]:.4f}"
                })
        
        # AUC metrics
        if 'macro_auc' in metrics:
            summary_data.append({
                'Metric': 'Macro AUC',
                'Value': f"{metrics['macro_auc']:.4f}"
            })
        
        if 'auc_ovr' in metrics:
            summary_data.append({
                'Metric': 'AUC (One-vs-Rest)',
                'Value': f"{metrics['auc_ovr']:.4f}"
            })
        
        return pd.DataFrame(summary_data)


class ModelEvaluator:
    """Complete model evaluation pipeline"""
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.metrics_calculator = DRMetrics()
        self.visualizer = MultiModalVisualizer(model, device)
    
    def evaluate_model(
        self, 
        dataloader,
        save_dir: Optional[str] = None
    ) -> Dict[str, any]:
        """Complete model evaluation"""
        
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_patient_ids = []
        
        # Collect predictions
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Collect results
                probs = torch.softmax(outputs['logits'], dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                all_labels.extend(batch['label'].cpu().numpy())
                
                if 'patient_id' in batch:
                    all_patient_ids.extend(batch['patient_id'])
        
        # Convert to arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_probs = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(y_true, y_pred, y_probs)
        
        # Create visualizations
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Confusion matrix
            self.metrics_calculator.plot_confusion_matrix(
                y_true, y_pred, save_path=save_dir / 'confusion_matrix.png'
            )
            
            # Classification report
            self.metrics_calculator.plot_classification_report(
                y_true, y_pred, save_path=save_dir / 'classification_report.png'
            )
            
            # Save metrics summary
            summary_df = self.metrics_calculator.create_metrics_summary(metrics)
            summary_df.to_csv(save_dir / 'metrics_summary.csv', index=False)
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_probs,
            'true_labels': y_true,
            'patient_ids': all_patient_ids
        }


print("Visualization and metrics modules completed!")
print("\nProject structure is now complete with:")
print("✓ Data pipeline (preprocessing, transforms, dataset)")
print("✓ Model architectures (SSL, fusion, foundation models)")
print("✓ Training utilities and loss functions")
print("✓ Visualization and explainability tools")
print("✓ Comprehensive evaluation metrics")
print("\nNext: Create Jupyter notebooks for experimentation!")
