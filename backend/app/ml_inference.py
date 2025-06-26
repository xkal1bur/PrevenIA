# ML inference service for genetic variant prediction using real trained models

from typing import Dict, Any, Optional
import random
import os
import pickle
from pathlib import Path
import numpy as np
import boto3
import math

class ModelInferenceService:
    """Servicio de inferencia ML que devuelve predicciones realistas basadas en modelos entrenados"""
    
    def __init__(self):
        # Directorio donde se guardaron los modelos entrenados
        self.models_dir = Path(__file__).resolve().parent / "models" / "ml_results" / "models"
        self.trained_models = {}
        self.pca_transformer = None
        self.scaler = None
        
        # Configuración S3 para buscar embeddings de pacientes
        self.s3_client = None
        self.bucket_name = None

        # Cliente S3 opcional para embeddings
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                region_name=os.getenv("AWS_REGION", "us-east-1"),
            )
            self.bucket_name = os.getenv("S3_BUCKET_NAME")
        except Exception:
            self.s3_client = None
            self.bucket_name = None

        # Cargar modelos entrenados (tanto original como PCA)
        self._load_trained_models()

    def get_predictions_for_patient(self, dni: str, nombres: str, apellidos: str) -> Dict[str, Any]:
        """Obtiene predicciones de los modelos entrenados.

        Si no se encuentra un embedding válido o no hay modelos cargados,
        retorna un error explícito.
        """

        # 1) Verificar que tenemos modelos cargados y un embedding disponible
        embedding = self._load_patient_embedding(dni)

        # Validar embedding (debe tener 32768 características)
        if embedding is None:
            print(f"[ModelInference] ❌ Embedding es None")
        else:
            print(f"[ModelInference] 📐 Embedding shape: {embedding.shape}")
            print(f"[ModelInference] 🔢 Características esperadas: 32768, encontradas: {embedding.shape[0] if len(embedding.shape) > 0 else 'N/A'}")
            
        if embedding is None or embedding.shape[0] != 32768:
            print(f"[ModelInference] ⚠️ Embedding inválido para paciente {dni}. No se puede realizar predicción.")
            return {
                "status": "error",
                "total_models": 0,
                "patient_info": {
                    "dni": dni,
                    "name": f"{nombres} {apellidos}"
                },
                "sample_used": "No disponible",
                "predictions": {},
                "description": f"No se encontró un embedding válido para el paciente {nombres} {apellidos} (DNI: {dni})"
            }

        if embedding is not None and embedding.shape[0] == 32768:
            print(f"[ModelInference] ✅ Embedding válido encontrado, procediendo con predicciones...")
            try:
                predictions = {}
                pathogenic_count = 0
                benign_count = 0
                probs_sum = 0.0

                for model_name, model in self.trained_models.items():
                    model_obj = model["model"]
                    representation = model["representation"]
                    
                    # Preparar los datos según la representación del modelo
                    if representation == "original":
                        # Usar embedding directamente (ya debe estar escalado en el proceso de generación)
                        X = embedding.reshape(1, -1)
                    elif representation == "pca":
                        if self.pca_transformer is None:
                            print(f"[ModelInference] ⚠️ Saltando modelo PCA {model_name} (sin transformer)")
                            continue
                        # Aplicar scaling y luego PCA
                        if self.scaler:
                            X_scaled = self.scaler.transform(embedding.reshape(1, -1))
                        else:
                            X_scaled = embedding.reshape(1, -1)  # Asumir ya escalado
                        X = self.pca_transformer.transform(X_scaled)
                    else:
                        print(f"[ModelInference] ⚠️ Representación desconocida: {representation}")
                        continue

                    # Obtener probabilidad de clase positiva (1 = Pathogenic)
                    if hasattr(model_obj, "predict_proba"):
                        prob_pathogenic = float(model_obj.predict_proba(X)[0][1])
                    elif hasattr(model_obj, "decision_function"):
                        # Escalar decision_function a [0,1] con sigmoide
                        prob_pathogenic = float(1 / (1 + math.exp(-model_obj.decision_function(X)[0])))
                    else:
                        # Predict devuelve 0/1
                        prob_pathogenic = float(model_obj.predict(X)[0])

                    prediction_label = "Pathogenic" if prob_pathogenic >= 0.5 else "Benign"
                    if prediction_label == "Pathogenic":
                        pathogenic_count += 1
                    else:
                        benign_count += 1

                    probs_sum += prob_pathogenic

                    # Asignar confianza heurística
                    if prob_pathogenic >= 0.85 or prob_pathogenic <= 0.15:
                        confidence = "Muy Alta"
                    elif prob_pathogenic >= 0.70 or prob_pathogenic <= 0.30:
                        confidence = "Alta"
                    elif prob_pathogenic >= 0.60 or prob_pathogenic <= 0.40:
                        confidence = "Media-Alta"
                    else:
                        confidence = "Media"

                    predictions[model_name] = {
                        "prediction": prediction_label,
                        "probability": round(prob_pathogenic, 4),
                        "confidence": confidence,
                        "description": "Modelo entrenado real",
                        "model_performance": "N/A"
                    }

                total_models = len(predictions)
                avg_probability = probs_sum / total_models if total_models > 0 else 0.0

                # Generar interpretación simple
                consensus = "Pathogenic" if pathogenic_count > benign_count else "Benign"
                scenario_stub = {
                    "scenario_name": "Predicción basada en modelos reales",
                    "risk_level": "Alto" if consensus == "Pathogenic" else "Bajo",
                    "clinical_significance": "Patogénica" if consensus == "Pathogenic" else "Benigna",
                    "consensus": consensus,
                }

                consensus_confidence = "Alta" if max(pathogenic_count, benign_count) >= (0.7 * total_models) else "Media"

                return {
                    "status": "success",
                    "total_models": total_models,
                    "patient_info": {
                        "dni": dni,
                        "name": f"{nombres} {apellidos}"
                    },
                    "sample_used": f"embedding_paciente_{dni}.pkl",
                    "scenario_info": {
                        **scenario_stub,
                        "consensus_confidence": consensus_confidence
                    },
                    "analysis_summary": {
                        "models_predicting_pathogenic": pathogenic_count,
                        "models_predicting_benign": benign_count,
                        "average_probability": round(avg_probability, 4),
                        "prediction_agreement": f"{max(pathogenic_count, benign_count)}/{total_models} modelos coinciden"
                    },
                    "clinical_recommendations": [],
                    "predictions": predictions,
                    "interpretation": f"Consenso \u2192 {consensus} ({consensus_confidence}) basado en {total_models} modelos.",
                    "description": f"Predicción automática a partir de embedding ML para {nombres} {apellidos} (DNI: {dni})"
                }
            except Exception as e:
                print(f"[ModelInference] Error durante inferencia real: {e}")
        else:
            print(f"[ModelInference] ❌ No hay modelos entrenados cargados")

        # Si falla (sin modelos o sin embedding) devolver error explícito
        return {
            "status": "error",
            "total_models": 0,
            "patient_info": {
                "dni": dni,
                "name": f"{nombres} {apellidos}"
            },
            "sample_used": "No disponible",
            "predictions": {},
            "description": f"No hay modelos entrenados cargados o no se encontró un embedding válido para el paciente {nombres} {apellidos} (DNI: {dni})"
        }
    
    def get_predictions_for_patient_with_embedding(self, dni: str, nombres: str, apellidos: str, embedding: np.ndarray, embedding_filename: str) -> Dict[str, Any]:
        """Obtiene predicciones de los modelos entrenados usando un embedding específico.

        Args:
            dni: DNI del paciente
            nombres: Nombres del paciente  
            apellidos: Apellidos del paciente
            embedding: Array numpy con el embedding (32768 características)
            embedding_filename: Nombre del archivo .pkl del embedding
        """
        
        print(f"[ModelInference] 🎯 Generando predicciones con embedding específico: {embedding_filename}")
        print(f"[ModelInference] 📐 Embedding shape: {embedding.shape}")
        print(f"[ModelInference] 🔢 Modelos disponibles: {len(self.trained_models)}")

        # Verificar que el embedding tiene la forma correcta
        if embedding.shape[0] != 32768:
            raise ValueError(f"Embedding inválido: esperado 32768 características, encontrado {embedding.shape[0]}")

        if len(self.trained_models) == 0:
            print(f"[ModelInference] ❌ No hay modelos entrenados cargados")
            return {
                "status": "error",
                "total_models": 0,
                "patient_info": {
                    "dni": dni,
                    "name": f"{nombres} {apellidos}"
                },
                "sample_used": embedding_filename,
                "predictions": {},
                "description": f"No hay modelos entrenados disponibles"
            }

        try:
            predictions = {}
            pathogenic_count = 0
            benign_count = 0
            probs_sum = 0.0

            print(f"[ModelInference] 🚀 Procesando con {len(self.trained_models)} modelos...")

            for model_name, model in self.trained_models.items():
                print(f"[ModelInference]   📊 Procesando modelo: {model_name}")
                model_obj = model["model"]
                representation = model["representation"]
                
                # Preparar los datos según la representación del modelo
                if representation == "original":
                    # Usar embedding directamente
                    X = embedding.reshape(1, -1)
                    print(f"[ModelInference]     📊 Usando embedding original: shape {X.shape}")
                elif representation == "pca":
                    if self.pca_transformer is None:
                        print(f"[ModelInference]   ⚠️ Saltando modelo PCA {model_name} (sin transformer)")
                        continue
                    
                    # Aplicar scaling y luego PCA
                    X_original = embedding.reshape(1, -1)
                    print(f"[ModelInference]     📊 Embedding original para PCA: shape {X_original.shape}")
                    
                    if self.scaler:
                        X_scaled = self.scaler.transform(X_original)
                        print(f"[ModelInference]     📊 Después de scaler: shape {X_scaled.shape}")
                    else:
                        X_scaled = X_original  # Asumir ya escalado
                        print(f"[ModelInference]     📊 Sin scaler, usando embedding directo: shape {X_scaled.shape}")
                    
                    X = self.pca_transformer.transform(X_scaled)
                    print(f"[ModelInference]     📊 Después de PCA: shape {X.shape}")
                    
                    # Verificar compatibilidad con el modelo
                    expected_features = getattr(model_obj, 'n_features_in_', None)
                    if expected_features and X.shape[1] != expected_features:
                        print(f"[ModelInference]     ❌ Incompatibilidad: modelo espera {expected_features} características, PCA produce {X.shape[1]}")
                        print(f"[ModelInference]     ⚠️ Saltando modelo PCA {model_name} (incompatible)")
                        continue
                    
                    print(f"[ModelInference]     ✅ Compatibilidad verificada: {X.shape[1]} características")
                else:
                    print(f"[ModelInference]   ⚠️ Representación desconocida: {representation}")
                    continue

                # Obtener probabilidad de clase positiva (1 = Pathogenic)
                if hasattr(model_obj, "predict_proba"):
                    prob_pathogenic = float(model_obj.predict_proba(X)[0][1])
                elif hasattr(model_obj, "decision_function"):
                    # Escalar decision_function a [0,1] con sigmoide
                    prob_pathogenic = float(1 / (1 + math.exp(-model_obj.decision_function(X)[0])))
                else:
                    # Predict devuelve 0/1
                    prob_pathogenic = float(model_obj.predict(X)[0])

                prediction_label = "Pathogenic" if prob_pathogenic >= 0.5 else "Benign"
                if prediction_label == "Pathogenic":
                    pathogenic_count += 1
                else:
                    benign_count += 1

                probs_sum += prob_pathogenic

                # Asignar confianza heurística
                if prob_pathogenic >= 0.85 or prob_pathogenic <= 0.15:
                    confidence = "Muy Alta"
                elif prob_pathogenic >= 0.70 or prob_pathogenic <= 0.30:
                    confidence = "Alta"
                elif prob_pathogenic >= 0.60 or prob_pathogenic <= 0.40:
                    confidence = "Media-Alta"
                else:
                    confidence = "Media"

                predictions[model_name] = {
                    "prediction": prediction_label,
                    "probability": round(prob_pathogenic, 4),
                    "confidence": confidence,
                    "description": "Modelo entrenado real con embedding específico",
                    "model_performance": "N/A"
                }

                print(f"[ModelInference]     ✅ {model_name}: {prediction_label} ({prob_pathogenic:.4f})")

            total_models = len(predictions)
            avg_probability = probs_sum / total_models if total_models > 0 else 0.0

            # Generar interpretación
            consensus = "Pathogenic" if pathogenic_count > benign_count else "Benign"
            consensus_confidence = "Alta" if max(pathogenic_count, benign_count) >= (0.7 * total_models) else "Media"

            print(f"[ModelInference] 🎯 Consenso: {consensus} ({pathogenic_count}/{total_models} pathogenic)")
            print(f"[ModelInference] 📊 Probabilidad promedio: {avg_probability:.4f}")

            return {
                "status": "success",
                "total_models": total_models,
                "patient_info": {
                    "dni": dni,
                    "name": f"{nombres} {apellidos}"
                },
                "sample_used": embedding_filename,
                "scenario_info": {
                    "scenario_name": f"Predicción con embedding específico: {embedding_filename}",
                    "risk_level": "Alto" if consensus == "Pathogenic" else "Bajo",
                    "clinical_significance": "Patogénica" if consensus == "Pathogenic" else "Benigna",
                    "consensus": consensus,
                    "consensus_confidence": consensus_confidence
                },
                "analysis_summary": {
                    "models_predicting_pathogenic": pathogenic_count,
                    "models_predicting_benign": benign_count,
                    "average_probability": round(avg_probability, 4),
                    "prediction_agreement": f"{max(pathogenic_count, benign_count)}/{total_models} modelos coinciden"
                },
                "clinical_recommendations": [
                    f"Análisis basado en embedding específico: {embedding_filename}",
                    "Consultar con especialista en genética médica" if consensus == "Pathogenic" else "Continuar seguimiento de rutina"
                ],
                "predictions": predictions,
                "interpretation": f"Consenso → {consensus} ({consensus_confidence}) basado en {total_models} modelos usando embedding específico.",
                "description": f"Predicción automática con embedding específico ({embedding_filename}) para {nombres} {apellidos} (DNI: {dni})"
            }

        except Exception as e:
            print(f"[ModelInference] ❌ Error durante inferencia con embedding específico: {e}")
            raise Exception(f"Error procesando con embedding específico: {str(e)}")
    
    # ------------------------------------------------------------------
    #   CARGA DE MODELOS ENTRENADOS
    # ------------------------------------------------------------------
    def _load_trained_models(self):
        """Carga los modelos .pk entrenados en self.models_dir.
        
        Carga tanto modelos 'original' como 'pca', junto con sus transformadores necesarios.
        """
        if not self.models_dir.exists():
            print(f"[ModelInference] ⚠️  Carpeta de modelos no encontrada: {self.models_dir}")
            return

        # Cargar transformadores (PCA, scaler)
        self.pca_transformer = None
        self.scaler = None
        
        # Intentar cargar el scaler global
        scaler_path = self.models_dir / "global_scaler.pk"
        if scaler_path.exists():
            try:
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                print(f"[ModelInference] ✅ Scaler global cargado")
            except Exception as e:
                print(f"[ModelInference] ❌ Error cargando scaler: {e}")

        for file in self.models_dir.glob("*.pk"):
            # Ignorar archivos de transformadores
            if file.name in ["global_scaler.pk", "pca_transformer.pk"]:
                continue

            # Formato guardado: rank-rep-model_name.pk
            parts = file.stem.split("-", 2)
            if len(parts) < 3:
                continue  # Nombre inesperado

            _, rep_part, model_part = parts

            try:
                with open(file, "rb") as f:
                    model_obj = pickle.load(f)

                display_name = model_part.replace("_", " ").title()
                # Incluir la representación en el nombre para evitar duplicados
                full_display_name = f"{display_name} ({rep_part.upper()})" if rep_part != "original" else display_name
                self.trained_models[full_display_name] = {
                    "model": model_obj,
                    "representation": rep_part
                }
                print(f"[ModelInference] ✅ Modelo cargado: {full_display_name} (rep: {rep_part})")
            except Exception as e:
                print(f"[ModelInference] ❌ Error cargando {file.name}: {e}")

        # Cargar el transformador PCA si hay modelos PCA
        pca_models = [name for name, info in self.trained_models.items() if info["representation"] == "pca"]
        if pca_models and not self.pca_transformer:
            # Intentar cargar PCA transformer guardado
            pca_path = self.models_dir / "pca_transformer.pk"
            if pca_path.exists():
                try:
                    with open(pca_path, "rb") as f:
                        self.pca_transformer = pickle.load(f)
                    print(f"[ModelInference] ✅ PCA transformer cargado desde archivo ({self.pca_transformer.n_components_} componentes)")
                except Exception as e:
                    print(f"[ModelInference] ❌ Error cargando PCA transformer: {e}")
            else:
                print(f"[ModelInference] ⚠️ PCA transformer no encontrado en {pca_path}")

    def _load_patient_embedding(self, dni: str):
        """Intenta cargar el embedding de un paciente.

        Busca primero un archivo local llamado `embedding_paciente_<dni>.pkl` en el
        directorio actual.  Si no lo encuentra y hay un cliente S3 configurado,
        busca el objeto dentro de la carpeta del paciente.<dni>/.
        Devuelve un array numpy 1-D o None si no se encuentra.
        """
        local_file = Path(f"embedding_paciente_{dni}.pkl")
        if local_file.exists():
            try:
                return pickle.load(f)
            except Exception as e:
                print(f"[ModelInference] Error leyendo embedding local: {e}")
                
        # Buscar en S3
        if self.s3_client and self.bucket_name:
            try:
                prefix = f"{dni}/"
                resp = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
                
                # Buscar el archivo de embedding más relevante
                embedding_files = []
                for obj in resp.get("Contents", []):
                    key = obj["Key"]
                    if key.endswith(".pkl") and "embedding" in key:
                        embedding_files.append((key, obj["Size"]))
                
                if not embedding_files:
                    print(f"[ModelInference] No se encontraron archivos de embedding para {dni}")
                    return None
                
                # Preferir archivos más específicos (que contengan el DNI) y más grandes (más datos)
                embedding_files.sort(key=lambda x: (dni in x[0], x[1]), reverse=True)
                selected_file = embedding_files[0][0]
                
                print(f"[ModelInference] Cargando embedding: {selected_file}")
                embedding_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=selected_file)
                embedding = pickle.loads(embedding_obj["Body"].read())
                print(f"[ModelInference] ✅ Embedding cargado. Shape: {embedding.shape}")
                return embedding
            except Exception as e:
                print(f"[ModelInference] Error descargando embedding de S3: {e}")
                
        return None

# Instancia global del servicio
ml_service = ModelInferenceService() 