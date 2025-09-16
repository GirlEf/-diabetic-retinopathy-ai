# ============================================================================
# /src/utils/visualize.py
# Visualization and Explainability Tools
# ============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import cv2
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Grad-CAM implementation
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class MultiModalVisualizer:
    """Comprehensive visualization tool for multi-modal DR analysis"""
    
    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
        self.model.eval()
        
        # Setup Grad-CAM for different modalities
        self.grad_cams = self._setup_grad_cams()
    
    def _setup_grad_cams(self) -> Dict[str, GradCAM]:
        """Setup Grad-CAM for each modality encoder"""
        grad_cams = {}
        
        # Fundus Grad-CAM (targeting last conv layer of encoder)
        if hasattr(self.model, 'fundus_encoder'):
            target_layers = [self.model.fundus_encoder.backbone.encoder.layer[-1]]
            grad_cams['fundus'] = GradCAM(
                model=self.model.fundus_encoder,
                target_layers=target_layers,
                use_cuda=torch.cuda.is_available()
            )
        
        # OCT Grad-CAM
        if hasattr(self.model, 'oct_encoder'):
            # Find last conv layer in the sequential model
            target_layers = []
            for layer in reversed(self.model.oct_encoder.backbone):
                if isinstance(layer, torch.nn.Conv3d):
                    target_layers = [layer]
                    break
            if target_layers:
                grad_cams['oct'] = GradCAM(
                    model=self.model.oct_encoder,
                    target_layers=target_layers,
                    use_cuda=torch.cuda.is_available()
                )
        
        # FLIO Grad-CAM
        if hasattr(self.model, 'flio_encoder'):
            target_layers = []
            for layer in reversed(self.model.flio_encoder.backbone):
                if isinstance(layer, torch.nn.Conv2d):
                    target_layers = [layer]
                    break
            if target_layers:
                grad_cams['flio'] = GradCAM(
                    model=self.model.flio_encoder,
                    target_layers=target_layers,
                    use_cuda=torch.cuda.is_available()
                )
        
        return grad_cams
    
    def generate_grad_cam_visualization(
        self,
        batch: Dict[str, torch.Tensor],
        target_class: Optional[int] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Generate Grad-CAM visualizations for all modalities"""
        
        visualizations = {}
        
        with torch.no_grad():
            # Get model prediction if target_class not specified
            if target_class is None:
                outputs = self.model(batch)
                target_class = torch.argmax(outputs['logits'], dim=1)[0].item()
        
        targets = [ClassifierOutputTarget(target_class)]
        
        # Generate Grad-CAM for each modality
        for modality in ['fundus', 'oct', 'flio']:
            if modality in batch and modality in self.grad_cams:
                
                # Prepare input
                input_tensor = batch[modality]
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                
                # Generate Grad-CAM
                cam = self.grad_cams[modality](
                    input_tensor=input_tensor,
                    targets=targets
                )
                
                # Convert to visualization
                if modality == 'fundus':
                    vis = self._create_fundus_visualization(
                        input_tensor[0], cam[0]
                    )
                elif modality == 'oct':
                    vis = self._create_oct_visualization(
                        input_tensor[0], cam[0]
                    )
                elif modality == 'flio':
                    vis = self._create_flio_visualization(
                        input_tensor[0], cam[0]
                    )
                
                visualizations[modality] = vis
        
        # Create combined visualization
        if len(visualizations) > 1:
            combined_vis = self._create_combined_visualization(
                visualizations, target_class
            )
            
            if save_path:
                self._save_visualization(combined_vis, save_path)
        
        return visualizations
    
    def _create_fundus_visualization(
        self, 
        image: torch.Tensor, 
        cam: np.ndarray
    ) -> np.ndarray:
        """Create fundus Grad-CAM visualization"""
        # Convert tensor to numpy and normalize
        if image.dim() == 3:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        # Normalize image to [0, 1]
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        
        # Create visualization
        visualization = show_cam_on_image(img_np, cam_resized, use_rgb=True)
        
        return visualization
    
    def _create_oct_visualization(
        self, 
        volume: torch.Tensor, 
        cam: np.ndarray
    ) -> np.ndarray:
        """Create OCT volume Grad-CAM visualization"""
        # Take middle slice for visualization
        if volume.dim() == 4:  # [C, D, H, W]
            middle_slice_idx = volume.shape[1] // 2
            slice_2d = volume[0, middle_slice_idx].cpu().numpy()
        else:
            slice_2d = volume.cpu().numpy()
        
        # Normalize slice
        slice_2d = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
        
        # Convert to RGB for visualization
        slice_rgb = np.stack([slice_2d] * 3, axis=-1)
        
        # Resize CAM and apply
        cam_resized = cv2.resize(cam, (slice_2d.shape[1], slice_2d.shape[0]))
        visualization = show_cam_on_image(slice_rgb, cam_resized, use_rgb=True)
        
        return visualization
    
    def _create_flio_visualization(
        self, 
        flio_data: torch.Tensor, 
        cam: np.ndarray
    ) -> np.ndarray:
        """Create FLIO Grad-CAM visualization"""
        # Use intensity channels for base image
        if flio_data.shape[0] >= 4:  # [C, H, W]
            # Combine intensity channels
            intensity_ch1 = flio_data[2].cpu().numpy()
            intensity_ch2 = flio_data[3].cpu().numpy()
            base_img = (intensity_ch1 + intensity_ch2) / 2
        else:
            base_img = flio_data[0].cpu().numpy()
        
        # Normalize
        base_img = (base_img - base_img.min()) / (base_img.max() - base_img.min())
        
        # Convert to RGB
        base_rgb = np.stack([base_img] * 3, axis=-1)
        
        # Apply CAM
        cam_resized = cv2.resize(cam, (base_img.shape[1], base_img.shape[0]))
        visualization = show_cam_on_image(base_rgb, cam_resized, use_rgb=True)
        
        return visualization
    
    def _create_combined_visualization(
        self, 
        visualizations: Dict[str, np.ndarray], 
        target_class: int
    ) -> np.ndarray:
        """Create combined multi-modal visualization"""
        fig, axes = plt.subplots(1, len(visualizations), figsize=(15, 5))
        
        if len(visualizations) == 1:
            axes = [axes]
        
        dr_grades = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
        
        for idx, (modality, vis) in enumerate(visualizations.items()):
            axes[idx].imshow(vis)
            axes[idx].set_title(f'{modality.upper()} - Prediction: {dr_grades[target_class]}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Convert to numpy array
        fig.canvas.draw()
        combined_vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        combined_vis = combined_vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return combined_vis
    
    def visualize_attention_weights(
        self, 
        batch: Dict[str, torch.Tensor],
        save_path: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """Visualize cross-modal attention weights"""
        
        with torch.no_grad():
            outputs = self.model(batch)
            
            if 'attention_weights' not in outputs:
                return {}
            
            attention_weights = outputs['attention_weights']
            
        # Create attention visualizations
        attention_plots = {}
        
        for attention_name, weights in attention_weights.items():
            if weights is not None:
                # Average across heads and batch
                if weights.dim() > 2:
                    weights_avg = weights.mean(dim=0).mean(dim=0)  # [seq_len, seq_len]
                else:
                    weights_avg = weights.mean(dim=0)
                
                # Create heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(
                    weights_avg.cpu().numpy(),
                    annot=True,
                    fmt='.2f',
                    cmap='Blues',
                    ax=ax
                )
                ax.set_title(f'Attention Weights: {attention_name}')
                
                # Convert to array
                fig.canvas.draw()
                attention_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                attention_plot = attention_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                
                attention_plots[attention_name] = attention_plot
                plt.close(fig)
        
        return attention_plots
    
    def create_feature_importance_plot(
        self, 
        batch: Dict[str, torch.Tensor],
        class_names: List[str] = None
    ) -> go.Figure:
        """Create interactive feature importance plot"""
        
        if class_names is None:
            class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
        
        with torch.no_grad():
            outputs = self.model(batch)
            
            if 'feature_importance' not in outputs:
                return None
            
            importance = outputs['feature_importance']
        
        # Create subplot for each modality
        modalities = list(importance.keys())
        fig = make_subplots(
            rows=1, 
            cols=len(modalities),
            subplot_titles=[f'{mod.upper()} Features' for mod in modalities],
            specs=[[{'type': 'bar'}] * len(modalities)]
        )
        
        colors = px.colors.qualitative.Set1
        
        for idx, (modality, imp_values) in enumerate(importance.items()):
            # Get top features
            imp_np = imp_values[0].cpu().numpy()  # First sample
            top_indices = np.argsort(imp_np)[-10:]  # Top 10 features
            
            fig.add_trace(
                go.Bar(
                    x=imp_np[top_indices],
                    y=[f'Feature {i}' for i in top_indices],
                    orientation='h',
                    name=modality.upper(),
                    marker_color=colors[idx % len(colors)]
                ),
                row=1, col=idx + 1
            )
        
        fig.update_layout(
            title='Feature Importance Across Modalities',
            showlegend=False,
            height=500
        )
        
        return fig
    
    def create_prediction_confidence_plot(
        self, 
        logits: torch.Tensor,
        true_labels: torch.Tensor,
        class_names: List[str] = None
    ) -> go.Figure:
        """Create prediction confidence visualization"""
        
        if class_names is None:
            class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR']
        
        # Calculate probabilities and confidence
        probs = F.softmax(logits, dim=1)
        confidence = torch.max(probs, dim=1)[0]
        predicted = torch.argmax(probs, dim=1)
        
        # Create scatter plot
        fig = go.Figure()
        
        for class_idx in range(len(class_names)):
            mask = true_labels == class_idx
            if mask.sum() > 0:
                fig.add_trace(go.Scatter(
                    x=predicted[mask].cpu().numpy(),
                    y=confidence[mask].cpu().numpy(),
                    mode='markers',
                    name=class_names[class_idx],
                    text=[f'True: {class_names[class_idx]}<br>Pred: {class_names[p]}<br>Conf: {c:.3f}'
                          for p, c in zip(predicted[mask].cpu().numpy(), confidence[mask].cpu().numpy())],
                    hovertemplate='%{text}<extra></extra>'
                ))
        
        fig.update_layout(
            title='Prediction Confidence by True Class',
            xaxis_title='Predicted Class',
            yaxis_title='Prediction Confidence',
            xaxis=dict(tickmode='array', tickvals=list(range(len(class_names))), ticktext=class_names)
        )
        
        return fig
    
    def _save_visualization(self, visualization: np.ndarray, save_path: str):
        """Save visualization to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(visualization)
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()

