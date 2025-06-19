# import torch  # Torch no longer required; kept commented to avoid dependency
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler

# Machine Learning Models
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    AdaBoostClassifier, VotingClassifier,
    GradientBoostingClassifier, HistGradientBoostingClassifier, BaggingClassifier
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

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost not available")
    XGB_AVAILABLE = False

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Progress bar
try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class MLModelEvaluator:
    def __init__(self, embeddings_path='final_embeddings.npy', labels_path='labels.npy', output_dir='ml_results', n_jobs=96):
        """Crea un evaluador usando embeddings y labels almacenados en archivos .npy.

        Parameters
        ----------
        embeddings_path : str
            Ruta al archivo NPY con los embeddings de forma (n_samples, n_features).
        labels_path : str
            Ruta al archivo NPY con las etiquetas binarias (n_samples,).
        output_dir : str
            Carpeta donde se guardarán los resultados y modelos.
        n_jobs : int
            Núcleo/threads para los modelos paralelizables.
        """
        self.embeddings_path = embeddings_path
        self.labels_path = labels_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.n_jobs = n_jobs  # number of parallel threads/cores
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        self.best_models = {}
        
    def load_and_preprocess_data(self):
        """Carga embeddings/labels desde NPY, realiza split y reducciones dimensionales."""
        print("="*60)
        print("CARGANDO Y PREPROCESANDO DATOS (.npy)")
        print("="*60)

        # ---------------------------------------------
        # 1. Cargar archivos NPY
        # ---------------------------------------------
        try:
            X = np.load(self.embeddings_path)
            y = np.load(self.labels_path)
        except Exception as e:
            print(f"❌ No se pudieron cargar los NPY: {e}")
            return False

        if y.ndim > 1:
            y = y.reshape(-1)

        if X.shape[0] != y.shape[0]:
            print(f"❌ Inconsistencia: {X.shape[0]} muestras en embeddings y {y.shape[0]} en labels")
            return False

        print(f"Shapes → X: {X.shape}, y: {y.shape}")

        # ---------------------------------------------
        # 2. Train/Test split (85/15) + stratify
        # ---------------------------------------------
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        # Guardar en la instancia para uso posterior (e.g., ensembles)
        self.X_train = X_train
        self.X_test = X_test

        print(f"Dimensión de características: {X_train.shape[1]}")

        # ------------------------------------------------------------------
        # 4. Limpieza de NaN / Inf
        # ------------------------------------------------------------------
        for name, arr in [('train', X_train), ('test', X_test)]:
            nan_mask = np.isnan(arr)
            inf_mask = np.isinf(arr)
            if nan_mask.any() or inf_mask.any():
                print(f"  ↳ {name}: reemplazando {nan_mask.sum()} NaN y {inf_mask.sum()} Inf por 0")
                arr[nan_mask | inf_mask] = 0

        # ------------------------------------------------------------------
        # 5. Dimensionality Reduction techniques
        # ------------------------------------------------------------------
        reducers = {}

        # Original (no reduction)
        reducers['Original'] = (X_train, X_test)

        # PCA
        print("Aplicando PCA (95% varianza)…")
        pca = PCA(n_components=0.95, random_state=42)
        reducers['PCA'] = (pca.fit_transform(X_train), pca.transform(X_test))

        # LDA (only if binary classification -> 1 comp)
        try:
            lda_components = min(len(np.unique(y_train))-1, 10)
            if lda_components >= 1:
                lda = LDA(n_components=lda_components)
                reducers['LDA'] = (lda.fit_transform(X_train, y_train), lda.transform(X_test))
        except Exception as e:
            print(f"LDA failed: {e}")

        self.reducers = reducers  # store for later

        # Store y and original for potential tree models
        self.y_train = y_train
        self.y_test = y_test

        print("Resumen reducción dimensional:")
        for key, (tr, te) in reducers.items():
            print(f"  {key}: {tr.shape[1]} características")

        return True
    
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
            'Deep MLP': MLPClassifier(
                hidden_layer_sizes=(512, 256, 128), random_state=42, max_iter=5000,
                early_stopping=True
            ),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Hist Gradient Boosting': HistGradientBoostingClassifier(random_state=42),
            'Bagging': BaggingClassifier(random_state=42)
        }
        
        # Add LightGBM if available
        if LGB_AVAILABLE:
            models['LightGBM'] = lgb.LGBMClassifier(
                random_state=42, verbose=-1
            )
        
        # Add XGBoost if available
        if XGB_AVAILABLE:
            models['XGBoost'] = XGBClassifier(
                random_state=42, use_label_encoder=False, eval_metric='logloss',
                n_estimators=200
            )
        
        return models
    
    def get_param_grids(self):
        """Return hyper-parameter grids for each model."""
        grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20, 50],
                'min_samples_split': [2, 5, 10]
            },
            'Extra Trees': {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20, 50]
            },
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10],
                'solver': ['liblinear']
            },
            'SVM (RBF)': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            },
            'SVM (Linear)': {
                'C': [0.1, 1, 10]
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 11, 21],
                'weights': ['uniform', 'distance']
            },
            'Decision Tree': {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'MLP Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (128, 64)],
                'alpha': [0.0001, 0.001, 0.01]
            },
            'Deep MLP': {
                'hidden_layer_sizes': [(512, 256, 128), (512, 256, 128, 64), (256, 256, 128)],
                'alpha': [0.0001, 0.001],
                'learning_rate_init': [0.001, 0.0005]
            },
            'AdaBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.5, 1.0, 1.5]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200, 400],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Hist Gradient Boosting': {
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [None, 10, 20],
                'max_iter': [100, 200]
            },
            'Bagging': {
                'n_estimators': [10, 50, 100],
                'max_samples': [0.5, 0.7, 1.0]
            }
        }
        if LGB_AVAILABLE:
            grids['LightGBM'] = {
                'n_estimators': [100, 500, 1000],
                'num_leaves': [31, 63, 127],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        if XGB_AVAILABLE:
            grids['XGBoost'] = {
                'n_estimators': [200, 400, 800],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 1.0]
            }
        return grids
    
    def evaluate_model(self, name, model, X_train, X_test, y_train, y_test, param_grid=None):
        """Evaluate a single model"""
        print(f"\nEvaluando {name}…")
        
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Timeout evaluating {name}")
            
            # Set timeout of 300 seconds (5 minutes) per model
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(100)
            
            # --------------------------------------------------------------
            #   Hyper-parameter optimisation (GridSearchCV)
            # --------------------------------------------------------------
            if param_grid:
                print("  ↳ Buscando mejores hiperparámetros… (GridSearchCV - métrica: PR AUC)")
                grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1,
                                    scoring='average_precision', verbose=1)
                grid.fit(X_train, y_train)
                model = grid.best_estimator_
                print(f"     Mejores parámetros: {grid.best_params_}")
            else:
                print(f"  Entrenando {name}…")
                model.fit(X_train, y_train)
            
            # Predictions
            print(f"  Generando predicciones...")
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Algunas implementaciones no tienen predict_proba
            if hasattr(model, 'predict_proba'):
                y_train_proba = model.predict_proba(X_train)[:, 1]
                y_test_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_train_proba = model.decision_function(X_train)
                y_test_proba = model.decision_function(X_test)
            else:
                y_train_proba = y_train_pred
                y_test_proba = y_test_pred
            
            # Metrics
            results = {
                'model_name': name,
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'train_auc': roc_auc_score(y_train, y_train_proba),
                'test_auc': roc_auc_score(y_test, y_test_proba),
                'train_pr_auc': average_precision_score(y_train, y_train_proba),
                'test_pr_auc': average_precision_score(y_test, y_test_proba),
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
            print(f"  Ejecutando validación cruzada (PR AUC)...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='average_precision', n_jobs=self.n_jobs)
            results['cv_auc_mean'] = cv_scores.mean()
            results['cv_auc_std'] = cv_scores.std()
            
            # Cancel timeout
            signal.alarm(0)
            
            print(f"  Test AUC: {results['test_auc']:.4f} | PR AUC: {results['test_pr_auc']:.4f}")
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
        
        # Iterate over each dimensionality reduction
        for red_name, (X_train_red, X_test_red) in self.reducers.items():
            print("\n" + "-"*60)
            print(f"USANDO REPRESENTACIÓN: {red_name}")
            print("-"*60)

            models = self.get_models()

            # Scale where beneficial
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_red)
            X_test_scaled = scaler.transform(X_test_red)

            scale_models = ['Logistic Regression', 'SVM (RBF)', 'SVM (Linear)', 
                            'K-Nearest Neighbors', 'MLP Neural Network', 'Deep MLP']

            iterator = models.items()
            if TQDM_AVAILABLE:
                iterator = tqdm(iterator, total=len(models), desc=f"Modelos ({red_name})")

            for name, model in iterator:
                full_name = f"{red_name} | {name}"

                # set n_jobs globally where supported
                if hasattr(model, 'n_jobs'):
                    try:
                        model.n_jobs = self.n_jobs
                    except Exception:
                        pass

                # Evaluate
                X_tr, X_te = (X_train_scaled, X_test_scaled) if name in scale_models else (X_train_red, X_test_red)

                results, trained_model = self.evaluate_model(
                    full_name, model, X_tr, X_te, self.y_train, self.y_test,
                    self.get_param_grids().get(name)
                )

                if results is not None:
                    self.results[full_name] = results
                    self.best_models[full_name] = {'model': trained_model, 'scaler': scaler if name in scale_models else None}
    
    def create_ensemble_models(self):
        """Create ensemble models from best performers"""
        print("\n" + "="*50)
        print("CREANDO MODELOS ENSEMBLE")
        print("="*50)
        
        if len(self.results) < 3:
            print("No hay suficientes modelos para crear ensemble")
            return
        
        # Get top 6 models by test PR AUC (better for imbalanced data)
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_pr_auc'], reverse=True)
        top_models = sorted_models[:6]
        
        print("Top 6 modelos para ensemble:")
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
                self.X_train, self.X_test, self.y_train, self.y_test,
                param_grid=None
            )
            
            if results_hard:
                self.results['Ensemble (Hard Voting)'] = results_hard
                self.best_models['Ensemble (Hard Voting)'] = {'model': model_hard, 'scaler': None}
            
            # Soft voting
            soft_voting = VotingClassifier(estimators=estimators, voting='soft')
            results_soft, model_soft = self.evaluate_model(
                'Ensemble (Soft Voting)', soft_voting,
                self.X_train, self.X_test, self.y_train, self.y_test,
                param_grid=None
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
        
        # Get top 6 models
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_pr_auc'], reverse=True)
        top_6 = sorted_models[:6]
        
        for rank, (name, results) in enumerate(top_6, 1):
            model_info = self.best_models[name]

            # Descomponer el nombre "Representación | Modelo"
            if '|' in name:
                rep_part, model_part = map(str.strip, name.split('|', 1))
            else:
                rep_part, model_part = 'Original', name.strip()

            # Generar nombre de archivo con formato: rank-rep-model_name.pk
            filename_base = f"{rank}-{rep_part.lower().replace(' ', '_')}-{model_part.lower().replace(' ', '_')}"
            filename_base = filename_base.replace('(', '').replace(')', '')

            # Guardar modelo
            model_file = models_dir / f"{filename_base}.pk"
            with open(model_file, 'wb') as f:
                pickle.dump(model_info['model'], f)

            # Guardar scaler si existe
            if model_info['scaler'] is not None:
                scaler_file = models_dir / f"{filename_base}-scaler.pk"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(model_info['scaler'], f)

            print(f"Modelo guardado: {model_file.name} (AUC: {results['test_auc']:.4f})")
    
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
        df_results = df_results.sort_values('test_pr_auc', ascending=False)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Test PR AUC comparison
        axes[0, 0].barh(df_results.index, df_results['test_pr_auc'])
        axes[0, 0].set_xlabel('Test PR AUC')
        axes[0, 0].set_title('Test PR AUC por Modelo')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Train vs Test PR AUC
        axes[0, 1].scatter(df_results['train_pr_auc'], df_results['test_pr_auc'], alpha=0.7)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0, 1].set_xlabel('Train PR AUC')
        axes[0, 1].set_ylabel('Test PR AUC')
        axes[0, 1].set_title('Train vs Test PR AUC')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add model names as annotations
        for i, (idx, row) in enumerate(df_results.iterrows()):
            axes[0, 1].annotate(idx[:10], (row['train_pr_auc'], row['test_pr_auc']), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Plot 3: Test metrics comparison
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_pr_auc']
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
        axes[1, 1].set_xlabel('Cross-Validation PR AUC')
        axes[1, 1].set_title('CV PR AUC ± Std (Top 10 Modelos)')
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
        
        # Sort by test PR AUC
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1]['test_pr_auc'], reverse=True)
        
        print(f"{'Rank':<4} {'Modelo':<25} {'Test PR AUC':<12} {'Test Acc':<10} {'Test F1':<10} {'CV AUC':<15}")
        print("-" * 80)
        
        for i, (name, results) in enumerate(sorted_models, 1):
            print(f"{i:<4} {name:<25} {results['test_pr_auc']:<12.4f} "
                  f"{results['test_accuracy']:<10.4f} {results['test_f1']:<10.4f} "
                  f"{results['cv_auc_mean']:.3f}±{results['cv_auc_std']:.3f}")
        
        # Best model details
        best_name, best_results = sorted_models[0]
        print(f"\n🏆 MEJOR MODELO: {best_name}")
        print(f"   Test PR AUC: {best_results['test_pr_auc']:.4f}")
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
    evaluator = MLModelEvaluator(
        embeddings_path='final_embeddings.npy',
        labels_path='labels.npy'
    )
    evaluator.run_complete_pipeline()


if __name__ == "__main__":
    main() 