import torch
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    print("LightGBM not available")
    LGB_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class MLModelEvaluator:
    def __init__(self, embeddings_path='brca1_embeddings.pth', 
                 data_path='41586_2018_461_MOESM3_ESM.xlsx', 
                 output_dir='ml_results'):
        self.embeddings_path = embeddings_path
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.best_models = {}
        
    def load_and_preprocess_data(self):
        """Load embeddings and BRCA1 data with same preprocessing as notebook"""
        print("="*60)
        print("CARGANDO Y PREPROCESANDO DATOS")
        print("="*60)
        
        # Load embeddings
        print("Cargando embeddings...")
        embeddings = torch.load(self.embeddings_path, map_location='cpu')
        print(f"Embeddings es un diccionario con {len(embeddings)} elementos")
        
        # Load BRCA1 data
        print("Cargando datos BRCA1...")
        brca1 = pd.read_excel(self.data_path, header=2)
        brca1 = brca1[[
            'chromosome', 'position (hg19)', 'reference', 'alt', 
            'function.score.mean', 'func.class'
        ]]
        
        brca1.rename(columns={
            'chromosome': 'chrom',
            'position (hg19)': 'pos',
            'reference': 'ref',
            'alt': 'alt',
            'function.score.mean': 'score',
            'func.class': 'class',
        }, inplace=True)
        
        brca1['class'] = brca1['class'].replace(['FUNC', 'INT'], 'FUNC/INT')
        brca1['target'] = brca1['class'].apply(lambda x: 1 if x == 'LOF' else 0)
        
        print(f"Shape de brca1: {brca1.shape}")
        
        # Verificar tipo de datos de embeddings
        sample_embedding = embeddings[list(embeddings.keys())[0]]
        print(f"Tipo de datos de embeddings: {sample_embedding.dtype}")
        
        # Align embeddings and targets
        brca1_reset = brca1.reset_index(drop=True)
        brca1_reset['idx'] = brca1_reset.index
        
        aligned_embeddings = []
        aligned_targets = []
        
        for idx in brca1_reset['idx']:
            if idx in embeddings:
                # Convert to float32 to avoid BFloat16 issues
                embedding_float32 = embeddings[idx].to(torch.float32)
                aligned_embeddings.append(embedding_float32)
                aligned_targets.append(brca1_reset.loc[idx, 'target'])
            else:
                print(f"Warning: idx {idx} no encontrado en embeddings")
        
        print(f"Número de muestras alineadas: {len(aligned_embeddings)}")
        
        if aligned_embeddings:
            # Stack all embeddings into a single tensor
            embeddings_tensor = torch.stack(aligned_embeddings)
            targets_tensor = torch.tensor(aligned_targets, dtype=torch.float32)
            
            print(f"Shape de embeddings_tensor: {embeddings_tensor.shape}")
            print(f"Tipo de datos final: {embeddings_tensor.dtype}")
            
            # Convert to numpy for sklearn
            X = embeddings_tensor.cpu().numpy()
            y = targets_tensor.cpu().numpy()
            
            # Check for NaN or infinite values
            print("Verificando datos...")
            nan_mask = np.isnan(X)
            inf_mask = np.isinf(X)
            if nan_mask.any():
                print(f"Warning: {nan_mask.sum()} valores NaN encontrados, reemplazando con 0")
                X[nan_mask] = 0
            if inf_mask.any():
                print(f"Warning: {inf_mask.sum()} valores infinitos encontrados, reemplazando con 0")
                X[inf_mask] = 0
            
            # Apply PCA for dimensionality reduction to prevent convergence issues
            print("Aplicando PCA para reducir dimensionalidad...")
            pca = PCA(n_components=0.95, random_state=42)  # Keep 95% of variance
            X_reduced = pca.fit_transform(X)
            print(f"Dimensionalidad reducida de {X.shape[1]} a {X_reduced.shape[1]} features")
            print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.4f}")
            
            # Store both original and reduced data
            self.X_original = X
            self.X_reduced = X_reduced
            self.pca = pca
            
            # Train/test split (80/20) - use reduced data by default
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_reduced, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Also create original data splits for tree-based models
            self.X_train_orig, self.X_test_orig, _, _ = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"Train set: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features (PCA)")
            print(f"Test set: {self.X_test.shape[0]} samples")
            print(f"Train - LOF: {self.y_train.sum()}, FUNC/INT: {(self.y_train == 0).sum()}")
            print(f"Test - LOF: {self.y_test.sum()}, FUNC/INT: {(self.y_test == 0).sum()}")
            
            return True
        else:
            print("Error: No se pudieron alinear embeddings y targets")
            return False
    
    def get_models(self):
        """Define all models to test"""
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=10000, solver='liblinear', C=0.1
            ),
            'SVM (RBF)': SVC(
                probability=True, random_state=42
            ),
            'SVM (Linear)': SVC(
                kernel='linear', probability=True, random_state=42
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Ridge Classifier': RidgeClassifier(random_state=42),
            'MLP Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000
            ),
            'AdaBoost': AdaBoostClassifier(random_state=42)
        }
        
        # Add LightGBM if available
        if LGB_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                random_state=42, verbose=-1
            )
        
        return models
    
    def evaluate_model(self, name, model, X_train, X_test, y_train, y_test):
        """Evaluate a single model"""
        print(f"\nEvaluando {name}...")
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Timeout evaluating {name}")
            
            # Set timeout of 300 seconds (5 minutes) per model
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(100)
            
            # Train model
            print(f"  Entrenando {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            print(f"  Generando predicciones...")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            results = {
                'model_name': name,
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'train_auc': roc_auc_score(y_train, y_train_proba),
                'test_auc': roc_auc_score(y_test, y_test_proba),
            }
            
            # Test set detailed metrics
            test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
                y_test, y_test_pred, average='binary'
            )
            
            results.update({
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            })
            
            # Cross-validation score (reduced CV to speed up)
            print(f"  Ejecutando validación cruzada...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
            results['cv_auc_mean'] = cv_scores.mean()
            results['cv_auc_std'] = cv_scores.std()
            
            # Cancel timeout
            signal.alarm(0)
            
            print(f"  Test AUC: {results['test_auc']:.4f}")
            print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
            print(f"  CV AUC: {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}")
            
            return results, model
            
        except TimeoutError as e:
            signal.alarm(0)
            print(f"  ⏰ Timeout: {name} tardó demasiado, saltando...")
            return None, None
        except Exception as e:
            signal.alarm(0)
            print(f"  Error evaluating {name}: {str(e)}")
            return None, None
    
    def run_evaluation(self):
        """Run evaluation on all models"""
        print("\n" + "="*60)
        print("EVALUANDO MODELOS DE MACHINE LEARNING")
        print("="*60)
        
        models = self.get_models()
        
        # Optional: Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # Models that benefit from scaling (use PCA reduced data)
        scale_models = ['Logistic Regression', 'SVM (RBF)', 'SVM (Linear)', 
                      'K-Nearest Neighbors', 'MLP Neural Network']
        
        # Tree-based models that work better with original high-dimensional data
        tree_models = ['Random Forest', 'Extra Trees', 'Decision Tree']
        
        for name, model in models.items():
            if name in scale_models:
                # Use scaled PCA data for these models
                results, trained_model = self.evaluate_model(
                    name, model, X_train_scaled, X_test_scaled, self.y_train, self.y_test
                )
                if results is not None:
                    self.results[name] = results
                    self.best_models[name] = {'model': trained_model, 'scaler': scaler}
            elif name in tree_models:
                # Use original high-dimensional data for tree models
                results, trained_model = self.evaluate_model(
                    name, model, self.X_train_orig, self.X_test_orig, self.y_train, self.y_test
                )
                if results is not None:
                    self.results[name] = results
                    self.best_models[name] = {'model': trained_model, 'scaler': None}
            else:
                # Use PCA reduced data for other models
                results, trained_model = self.evaluate_model(
                    name, model, self.X_train, self.X_test, self.y_train, self.y_test
                )
                if results is not None:
                    self.results[name] = results
                    self.best_models[name] = {'model': trained_model, 'scaler': None}
    
    def create_ensemble_models(self):
        """Create ensemble models from best performers"""
        print("\n" + "="*50)
        print("CREANDO MODELOS ENSEMBLE")
        print("="*50)
        
        if len(self.results) < 3:
            print("No hay suficientes modelos para crear ensemble")
            return
        
        # Get top 5 models by test AUC
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_auc'], reverse=True)
        top_models = sorted_models[:5]
        
        print("Top 5 modelos para ensemble:")
        for name, results in top_models:
            print(f"  {name}: AUC = {results['test_auc']:.4f}")
        
        # Create voting classifier
        estimators = []
        for name, _ in top_models:
            model_info = self.best_models[name]
            if model_info['scaler'] is not None:
                # For scaled models, we'll need to handle this differently
                # For simplicity, we'll skip them in ensemble or use them as-is
                continue
            estimators.append((name.replace(' ', '_'), model_info['model']))
        
        if len(estimators) >= 3:
            # Hard voting
            hard_voting = VotingClassifier(estimators=estimators, voting='hard')
            results_hard, model_hard = self.evaluate_model(
                'Ensemble (Hard Voting)', hard_voting, 
                self.X_train, self.X_test, self.y_train, self.y_test
            )
            
            if results_hard:
                self.results['Ensemble (Hard Voting)'] = results_hard
                self.best_models['Ensemble (Hard Voting)'] = {'model': model_hard, 'scaler': None}
            
            # Soft voting
            soft_voting = VotingClassifier(estimators=estimators, voting='soft')
            results_soft, model_soft = self.evaluate_model(
                'Ensemble (Soft Voting)', soft_voting,
                self.X_train, self.X_test, self.y_train, self.y_test
            )
            
            if results_soft:
                self.results['Ensemble (Soft Voting)'] = results_soft
                self.best_models['Ensemble (Soft Voting)'] = {'model': model_soft, 'scaler': None}
    
    def save_results(self):
        """Save results and best models"""
        print("\n" + "="*50)
        print("GUARDANDO RESULTADOS")
        print("="*50)
        
        # Save results to JSON
        results_file = self.output_dir / 'ml_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Resultados guardados en: {results_file}")
        
        # Save best models
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Get top 5 models
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_auc'], reverse=True)
        top_5 = sorted_models[:5]
        
        for name, results in top_5:
            model_info = self.best_models[name]
            safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
            
            # Save model
            model_file = models_dir / f'{safe_name}_model.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump(model_info['model'], f)
            
            # Save scaler if exists
            if model_info['scaler'] is not None:
                scaler_file = models_dir / f'{safe_name}_scaler.pkl'
                with open(scaler_file, 'wb') as f:
                    pickle.dump(model_info['scaler'], f)
            
            print(f"Modelo guardado: {safe_name} (AUC: {results['test_auc']:.4f})")
    
    def create_visualizations(self):
        """Create visualizations of results"""
        print("\n" + "="*50)
        print("CREANDO VISUALIZACIONES")
        print("="*50)
        
        if not self.results:
            print("No hay resultados para visualizar")
            return
        
        # Results DataFrame
        df_results = pd.DataFrame(self.results).T
        df_results = df_results.sort_values('test_auc', ascending=False)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Test AUC comparison
        axes[0, 0].barh(df_results.index, df_results['test_auc'])
        axes[0, 0].set_xlabel('Test AUC')
        axes[0, 0].set_title('Test AUC por Modelo')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Train vs Test AUC
        axes[0, 1].scatter(df_results['train_auc'], df_results['test_auc'], alpha=0.7)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('Train AUC')
        axes[0, 1].set_ylabel('Test AUC')
        axes[0, 1].set_title('Train vs Test AUC')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add model names as annotations
        for i, (idx, row) in enumerate(df_results.iterrows()):
            axes[0, 1].annotate(idx[:10], (row['train_auc'], row['test_auc']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 3: Test metrics comparison
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        df_metrics = df_results[metrics].head(10)  # Top 10 models
        
        x = np.arange(len(df_metrics.index))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            axes[1, 0].bar(x + i*width, df_metrics[metric], width, 
                          label=metric.replace('test_', '').title())
        
        axes[1, 0].set_xlabel('Modelos')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Métricas de Test (Top 10 Modelos)')
        axes[1, 0].set_xticks(x + width * 1.5)
        axes[1, 0].set_xticklabels([name[:10] for name in df_metrics.index], rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: CV AUC with error bars
        top_10 = df_results.head(10)
        axes[1, 1].barh(range(len(top_10)), top_10['cv_auc_mean'], 
                       xerr=top_10['cv_auc_std'], alpha=0.7)
        axes[1, 1].set_yticks(range(len(top_10)))
        axes[1, 1].set_yticklabels([name[:15] for name in top_10.index])
        axes[1, 1].set_xlabel('Cross-Validation AUC')
        axes[1, 1].set_title('CV AUC ± Std (Top 10 Modelos)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.output_dir / 'ml_results_comparison.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Gráficos guardados en: {plot_file}")
        plt.show()
    
    def print_summary(self):
        """Print summary of results"""
        print("\n" + "="*60)
        print("RESUMEN DE RESULTADOS")
        print("="*60)
        
        if not self.results:
            print("No hay resultados disponibles")
            return
        
        # Sort by test AUC
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_auc'], reverse=True)
        
        print(f"{'Rank':<4} {'Modelo':<25} {'Test AUC':<10} {'Test Acc':<10} {'Test F1':<10} {'CV AUC':<15}")
        print("-" * 80)
        
        for i, (name, results) in enumerate(sorted_models, 1):
            print(f"{i:<4} {name:<25} {results['test_auc']:<10.4f} "
                  f"{results['test_accuracy']:<10.4f} {results['test_f1']:<10.4f} "
                  f"{results['cv_auc_mean']:.3f}±{results['cv_auc_std']:.3f}")
        
        # Best model details
        best_name, best_results = sorted_models[0]
        print(f"\n🏆 MEJOR MODELO: {best_name}")
        print(f"   Test AUC: {best_results['test_auc']:.4f}")
        print(f"   Test Accuracy: {best_results['test_accuracy']:.4f}")
        print(f"   Test Precision: {best_results['test_precision']:.4f}")
        print(f"   Test Recall: {best_results['test_recall']:.4f}")
        print(f"   Test F1-Score: {best_results['test_f1']:.4f}")
        print(f"   CV AUC: {best_results['cv_auc_mean']:.4f} ± {best_results['cv_auc_std']:.4f}")
    
    def run_complete_pipeline(self):
        """Run the complete ML evaluation pipeline"""
        print("🚀 INICIANDO PIPELINE DE EVALUACIÓN DE MACHINE LEARNING")
        print("="*80)
        
        # Load and preprocess data
        if not self.load_and_preprocess_data():
            print("❌ Error en la carga de datos. Terminando...")
            return
        
        # Run model evaluation
        self.run_evaluation()
        
        # Create ensemble models
        self.create_ensemble_models()
        
        # Print summary
        self.print_summary()
        
        # Save results and models
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
        
        print("\n✅ PIPELINE COMPLETADO")
        print(f"📁 Resultados guardados en: {self.output_dir}")


def main():
    """Main function to run the ML evaluation"""
    evaluator = MLModelEvaluator()
    evaluator.run_complete_pipeline()


if __name__ == "__main__":
    main() 