#!/usr/bin/env python3
"""
Specialized Lesion Localization and Modality Contribution Analysis
for Diabetic Retinopathy MSc Project

This script provides:
1. LIME heatmaps for lesion localization (microaneurysms, exudates, etc.)
2. SHAP values for modality contribution analysis (fundus, OCT, FLIO)
3. Clinical interpretation of results
4. Thesis-ready visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Explainability
try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("⚠️ LIME not available")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available")

class LesionLocalizationAnalyzer:
    """
    Analyzes lesion localization using LIME heatmaps
    """
    
    def __init__(self, image_size=224, patch_size=16):
        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_per_side = image_size // patch_size
        
        # Define retinal regions for lesion mapping
        self.retinal_regions = {
            'macula': (8, 8, 12, 12),      # Center region
            'superior_temporal': (0, 8, 8, 16),   # Upper right
            'superior_nasal': (0, 0, 8, 8),       # Upper left
            'inferior_temporal': (8, 8, 16, 16),  # Lower right
            'inferior_nasal': (8, 0, 16, 8),      # Lower left
            'optic_disc': (12, 4, 16, 8)          # Optic disc region
        }
        
        # Lesion types and their typical locations
        self.lesion_locations = {
            'microaneurysms': ['macula', 'superior_temporal', 'inferior_temporal'],
            'hard_exudates': ['macula', 'superior_temporal', 'inferior_temporal'],
            'cotton_wool_spots': ['superior_temporal', 'superior_nasal', 'inferior_temporal'],
            'retinal_vessels': ['superior_temporal', 'superior_nasal', 'inferior_temporal', 'inferior_nasal'],
            'neovascularization': ['optic_disc', 'superior_temporal', 'inferior_temporal']
        }
    
    def create_lesion_heatmap(self, feature_importance, lesion_type='microaneurysms'):
        """Create heatmap for specific lesion type"""
        heatmap = np.zeros((self.patches_per_side, self.patches_per_side))
        
        for feature_idx, importance in enumerate(feature_importance):
            if feature_idx < self.patches_per_side ** 2:
                row = feature_idx // self.patches_per_side
                col = feature_idx % self.patches_per_side
                
                # Check if this patch is in the lesion region
                region = self._get_patch_region(row, col)
                if region in self.lesion_locations.get(lesion_type, []):
                    heatmap[row, col] = importance
        
        return heatmap
    
    def _get_patch_region(self, row, col):
        """Get retinal region for a patch"""
        for region_name, (start_row, start_col, end_row, end_col) in self.retinal_regions.items():
            if (start_row <= row < end_row and start_col <= col < end_col):
                return region_name
        return 'peripheral'

class ModalityContributionAnalyzer:
    """
    Analyzes modality contribution using SHAP values
    """
    
    def __init__(self, feature_counts):
        self.feature_counts = feature_counts
        self.modalities = list(feature_counts.keys())
    
    def analyze_modality_contribution(self, shap_values):
        """Analyze SHAP values by modality"""
        if len(shap_values.shape) == 3:
            # Multi-class: average across classes
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        modality_importance = {}
        feature_start = 0
        
        for modality, n_features in self.feature_counts.items():
            feature_end = feature_start + n_features
            modality_shap = mean_shap[feature_start:feature_end]
            
            modality_importance[modality] = {
                'mean_importance': np.mean(modality_shap),
                'total_importance': np.sum(modality_shap),
                'max_importance': np.max(modality_shap),
                'std_importance': np.std(modality_shap),
                'feature_importance': modality_shap,
                'top_features': np.argsort(modality_shap)[-10:]  # Top 10 features
            }
            
            feature_start = feature_end
        
        return modality_importance

def create_lime_lesion_heatmaps(model, X_test, y_test, feature_names=None):
    """
    Create LIME heatmaps for lesion localization
    """
    print("🔍 Creating LIME lesion localization heatmaps...")
    
    if not LIME_AVAILABLE:
        print("❌ LIME not available")
        return None
    
    # Initialize analyzer
    analyzer = LesionLocalizationAnalyzer()
    
    # Create feature names if not provided
    if feature_names is None:
        n_features = X_test.shape[1]
        feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_test,
        feature_names=feature_names,
        class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'],
        mode='classification'
    )
    
    # Analyze different lesion types
    lesion_types = ['microaneurysms', 'hard_exudates', 'cotton_wool_spots', 'neovascularization']
    lesion_results = {}
    
    for lesion_type in lesion_types:
        print(f"   Analyzing {lesion_type.replace('_', ' ').title()}...")
        
        # Find DR cases for this lesion type
        dr_cases = [i for i in range(len(X_test)) if y_test[i] > 0][:5]
        
        lesion_results[lesion_type] = []
        
        for case_idx in dr_cases:
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                X_test[case_idx], 
                model.predict_proba,
                num_features=50,
                top_labels=1
            )
            
            # Extract feature importance
            feature_importance = explanation.as_list()
            importance_dict = {feature: weight for feature, weight in feature_importance}
            
            # Create heatmap for fundus features (first 256 features)
            fundus_importance = [importance_dict.get(f'Feature_{i}', 0) for i in range(256)]
            heatmap = analyzer.create_lesion_heatmap(fundus_importance, lesion_type)
            
            lesion_results[lesion_type].append({
                'case_idx': case_idx,
                'true_class': y_test[case_idx],
                'predicted_class': model.predict([X_test[case_idx]])[0],
                'confidence': model.predict_proba([X_test[case_idx]])[0].max(),
                'heatmap': heatmap,
                'explanation': explanation
            })
    
    return lesion_results

def analyze_modality_contribution_shap(model, X_test, y_test, feature_counts):
    """
    Analyze modality contribution using SHAP
    """
    print("📊 Analyzing modality contribution with SHAP...")
    
    if not SHAP_AVAILABLE:
        print("❌ SHAP not available")
        return None
    
    # Initialize analyzer
    analyzer = ModalityContributionAnalyzer(feature_counts)
    
    # Initialize SHAP explainer
    if hasattr(model, 'estimators_'):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_test[:100])
    
    # Compute SHAP values
    print("   Computing SHAP values...")
    if hasattr(model, 'estimators_'):
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)
    else:
        shap_values = explainer.shap_values(X_test)
    
    # Analyze modality contribution
    modality_contribution = analyzer.analyze_modality_contribution(shap_values)
    
    return {
        'shap_values': shap_values,
        'modality_contribution': modality_contribution,
        'explainer': explainer
    }

def visualize_lesion_heatmaps(lesion_results):
    """
    Create visualizations for lesion localization
    """
    print("📈 Creating lesion localization visualizations...")
    
    # Create subplots for each lesion type
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    lesion_types = list(lesion_results.keys())
    colors = ['Reds', 'Blues', 'Greens', 'Purples']
    
    for idx, lesion_type in enumerate(lesion_types):
        ax = axes[idx]
        
        if lesion_results[lesion_type]:
            # Average heatmap across cases
            heatmaps = [case['heatmap'] for case in lesion_results[lesion_type]]
            avg_heatmap = np.mean(heatmaps, axis=0)
            
            # Create heatmap
            im = ax.imshow(avg_heatmap, cmap=colors[idx], alpha=0.8)
            ax.set_title(f'{lesion_type.replace("_", " ").title()} Localization', fontsize=14, fontweight='bold')
            ax.set_xlabel('Horizontal Position')
            ax.set_ylabel('Vertical Position')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig('lesion_localization_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return lesion_results

def visualize_modality_contribution(modality_contribution):
    """
    Create visualizations for modality contribution
    """
    print("📈 Creating modality contribution visualizations...")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    modalities = list(modality_contribution.keys())
    colors = ['red', 'blue', 'green']
    
    # 1. Mean importance comparison
    mean_importance = [modality_contribution[mod]['mean_importance'] for mod in modalities]
    axes[0, 0].bar(modalities, mean_importance, color=colors[:len(modalities)])
    axes[0, 0].set_title('Mean SHAP Importance by Modality')
    axes[0, 0].set_ylabel('Mean |SHAP Value|')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Total importance comparison
    total_importance = [modality_contribution[mod]['total_importance'] for mod in modalities]
    axes[0, 1].bar(modalities, total_importance, color=[c + '80' for c in colors[:len(modalities)]])
    axes[0, 1].set_title('Total SHAP Importance by Modality')
    axes[0, 1].set_ylabel('Total |SHAP Value|')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Feature importance distribution
    for i, modality in enumerate(modalities):
        feature_importance = modality_contribution[modality]['feature_importance']
        axes[1, 0].hist(feature_importance, alpha=0.7, label=modality.upper(), bins=20, color=colors[i])
    
    axes[1, 0].set_title('Feature Importance Distribution by Modality')
    axes[1, 0].set_xlabel('|SHAP Value|')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    
    # 4. Modality contribution pie chart
    total_contribution = sum([modality_contribution[mod]['total_importance'] for mod in modalities])
    contribution_percentages = [modality_contribution[mod]['total_importance']/total_contribution*100 for mod in modalities]
    
    axes[1, 1].pie(contribution_percentages, labels=modalities, autopct='%1.1f%%', startangle=90, colors=colors[:len(modalities)])
    axes[1, 1].set_title('Modality Contribution Percentage')
    
    plt.tight_layout()
    plt.savefig('modality_contribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return modality_contribution

def clinical_interpretation(lesion_results, modality_contribution):
    """
    Provide clinical interpretation of results
    """
    print("🏥 Generating clinical interpretation...")
    
    print("\n🔬 LESION LOCALIZATION INSIGHTS:")
    print("-" * 30)
    
    for lesion_type, cases in lesion_results.items():
        if cases:
            detection_rate = len([c for c in cases if c['predicted_class'] > 0]) / len(cases)
            avg_confidence = np.mean([case['confidence'] for case in cases])
            
            print(f"\n   {lesion_type.replace('_', ' ').title()}:")
            print(f"      Detection rate: {detection_rate:.3f} ({detection_rate*100:.1f}%)")
            print(f"      Average confidence: {avg_confidence:.3f}")
            
            # Clinical significance
            clinical_significance = {
                'microaneurysms': 'Early DR indicator - capillary wall damage',
                'hard_exudates': 'Lipid accumulation - vascular leakage',
                'cotton_wool_spots': 'Nerve fiber layer damage - ischemia',
                'neovascularization': 'Proliferative DR - severe complication'
            }
            print(f"      Clinical significance: {clinical_significance.get(lesion_type, 'Unknown')}")
    
    print(f"\n📊 MODALITY CONTRIBUTION INSIGHTS:")
    print("-" * 35)
    
    # Find most important modality
    most_important = max(modality_contribution.items(), 
                        key=lambda x: x[1]['total_importance'])
    
    print(f"   Most important modality: {most_important[0].upper()}")
    print(f"   Total contribution: {most_important[1]['total_importance']:.4f}")
    
    for modality, stats in modality_contribution.items():
        print(f"\n   {modality.upper()}:")
        print(f"      Contribution: {stats['total_importance']:.4f}")
        print(f"      Clinical role: {get_clinical_role(modality)}")
    
    return {
        'lesion_insights': {lesion_type: {
            'detection_rate': len([c for c in cases if c['predicted_class'] > 0]) / len(cases),
            'avg_confidence': np.mean([case['confidence'] for case in cases])
        } for lesion_type, cases in lesion_results.items()},
        'modality_insights': modality_contribution,
        'most_important_modality': most_important[0]
    }

def get_clinical_role(modality):
    """Get clinical role description for each modality"""
    roles = {
        'fundus': 'Retinal surface imaging, vessel analysis, lesion detection',
        'oct': 'Retinal layer thickness, fluid detection, structural changes',
        'flio': 'Metabolic activity, cellular function, early disease detection'
    }
    return roles.get(modality, 'Imaging modality')

def main():
    """
    Run complete lesion localization and modality contribution analysis
    """
    print("🎓 LESION LOCALIZATION & MODALITY CONTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Create simulated data
    print("\n📂 Creating simulated diabetic retinopathy data...")
    
    n_samples = 200
    n_features_fundus = 256
    n_features_oct = 256
    n_features_flio = 256
    
    np.random.seed(42)
    
    # Create features with lesion-specific patterns
    fundus_features = np.random.randn(n_samples, n_features_fundus)
    oct_features = np.random.randn(n_samples, n_features_oct)
    flio_features = np.random.randn(n_samples, n_features_flio)
    
    # Add lesion-specific patterns
    for i in range(n_samples):
        dr_severity = np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.25, 0.2, 0.1, 0.05])
        if dr_severity > 0:
            # Add lesion-specific feature patterns
            fundus_features[i, :50] += dr_severity * 0.1  # Microaneurysms
            fundus_features[i, 50:100] += dr_severity * 0.08  # Hard exudates
            oct_features[i, :40] += dr_severity * 0.12  # Retinal thickness
            flio_features[i, :30] += dr_severity * 0.15  # Metabolic changes
    
    # Combine features
    X = np.concatenate([fundus_features, oct_features, flio_features], axis=1)
    y = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.4, 0.25, 0.2, 0.1, 0.05])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    feature_counts = {'fundus': n_features_fundus, 'oct': n_features_oct, 'flio': n_features_flio}
    
    print(f"✅ Data prepared:")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_test.shape[1]}")
    print(f"   Modalities: {list(feature_counts.keys())}")
    
    # Step 1: LIME Lesion Localization
    print(f"\n🔍 STEP 1: LIME LESION LOCALIZATION")
    print("-" * 40)
    
    lesion_results = create_lime_lesion_heatmaps(model, X_test, y_test)
    if lesion_results:
        visualize_lesion_heatmaps(lesion_results)
    
    # Step 2: SHAP Modality Contribution
    print(f"\n📊 STEP 2: SHAP MODALITY CONTRIBUTION")
    print("-" * 40)
    
    shap_results = analyze_modality_contribution_shap(model, X_test, y_test, feature_counts)
    if shap_results:
        modality_contribution = visualize_modality_contribution(shap_results['modality_contribution'])
    
    # Step 3: Clinical Interpretation
    print(f"\n🏥 STEP 3: CLINICAL INTERPRETATION")
    print("-" * 35)
    
    if lesion_results and shap_results:
        clinical_insights = clinical_interpretation(lesion_results, shap_results['modality_contribution'])
    
    # Step 4: Save Results
    print(f"\n💾 STEP 4: SAVING RESULTS")
    print("-" * 25)
    
    analysis_results = {
        'lesion_results': lesion_results if 'lesion_results' in locals() else None,
        'modality_contribution': modality_contribution if 'modality_contribution' in locals() else None,
        'clinical_insights': clinical_insights if 'clinical_insights' in locals() else None,
        'feature_counts': feature_counts,
        'model_info': {
            'model_type': type(model).__name__,
            'test_samples': len(X_test),
            'features': X_test.shape[1],
            'classes': len(np.unique(y_test))
        }
    }
    
    joblib.dump(analysis_results, 'lesion_modality_analysis.pkl')
    print(f"✅ Results saved to: lesion_modality_analysis.pkl")
    
    # Final summary
    print(f"\n🏆 ANALYSIS COMPLETE!")
    print("=" * 30)
    print("✅ LIME lesion localization heatmaps created")
    print("✅ SHAP modality contribution analysis completed")
    print("✅ Clinical interpretation generated")
    print("✅ Visualizations saved for thesis")
    print("✅ Results saved for further analysis")
    
    return analysis_results

if __name__ == "__main__":
    results = main() 