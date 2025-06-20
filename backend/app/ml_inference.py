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
        self.models_dir = Path(__file__).resolve().parent / "models" / "ml_results_3" / "models"
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

        # ------------------------------------------------------------
        #  Cargar embedding por defecto (primera fila de target.csv)
        # ------------------------------------------------------------
        self.default_embedding: Optional[np.ndarray] = self._load_default_embedding()
        if self.default_embedding is None or self.default_embedding.shape[0] != 32768:
            print("[ModelInference] ⚠️ Default embedding could not be loaded or has invalid shape. ML predictions will be disabled.")

    def get_predictions_for_patient(self, dni: str, nombres: str, apellidos: str) -> Dict[str, Any]:
        """Obtiene predicciones de los modelos entrenados.

        Si no se encuentra un embedding válido o no hay modelos cargados,
        retrocede al comportamiento de escenarios simulados.
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
            print(f"[ModelInference] ⚠️ Embedding inválido para paciente {dni}. Usando embedding por defecto.")
            embedding = self.default_embedding

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

        # ------------------------------------------------------------------
        #   Si falla (sin modelos o sin embedding) devolver error explícito
        # ------------------------------------------------------------------
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
            # Ignorar el global_scaler
            if file.name == "global_scaler.pk":
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
            # Intentar recrear PCA desde los datos de entrenamiento
            try:
                embeddings_path = Path(__file__).resolve().parent / "models" / "final_embeddings.npy"
                if embeddings_path.exists():
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    
                    X = np.load(embeddings_path)
                    if self.scaler:
                        X_scaled = self.scaler.transform(X)
                    else:
                        scaler_temp = StandardScaler()
                        X_scaled = scaler_temp.fit_transform(X)
                    
                    pca = PCA(n_components=0.99, random_state=42)
                    pca.fit(X_scaled)
                    self.pca_transformer = pca
                    print(f"[ModelInference] ✅ PCA transformer creado ({pca.n_components_} componentes)")
                else:
                    print(f"[ModelInference] ⚠️ No se pudo cargar final_embeddings.npy para recrear PCA")
            except Exception as e:
                print(f"[ModelInference] ❌ Error creando PCA transformer: {e}")
                # Remover modelos PCA si no se puede cargar el transformer
                for name in pca_models:
                    del self.trained_models[name]
                    print(f"[ModelInference] ❌ Removido modelo PCA {name} (sin transformer)")

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

    def _load_default_embedding(self) -> Optional[np.ndarray]:
        """Carga la primera fila de target.csv como embedding por defecto.

        El archivo se espera en backend/app/models/target.csv. Se descarta la
        posible columna 'label' en la última posición.
        """
        target_path = (Path(__file__).resolve().parent / "models" / "target.csv").resolve()
        if not target_path.exists():
            print(f"[ModelInference] ⚠️ target.csv no encontrado en {target_path}")
            return None
        try:
            with open(target_path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            if len(lines) < 2:
                print("[ModelInference] ⚠️ target.csv no contiene datos suficientes")
                return None

            header = lines[0].split(",")
            values = lines[1].split(",")

            # Si la última cabecera es 'label' quitar la columna de las features
            if header[-1].lower() == "label" and len(values) == len(header):
                values = values[:-1]

            emb = np.asarray(values, dtype=np.float32)
            # Si la dimensión es 8192, replicar 4× para obtener 32768
            if emb.shape[0] == 8192:
                emb = np.tile(emb, 4)

            if emb.shape[0] != 32768:
                print(f"[ModelInference] ⚠️ Dimensión de default embedding inesperada: {emb.shape[0]} (esperado 32768)")
                return None
            return emb
        except Exception as e:
            print(f"[ModelInference] Error leyendo target.csv: {e}")
            return None

# Instancia global del servicio
ml_service = ModelInferenceService() 