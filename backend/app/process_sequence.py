import os
import io
import base64
import requests
import numpy as np
import zipfile
import json
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()
NIM_KEY = os.getenv("NVIDIA_NIM_KEY")

def extract_sequence_from_fasta(fasta_content: str) -> str:
    """Extrae la secuencia de un archivo FASTA, removiendo headers y espacios"""
    lines = fasta_content.split('\n')
    sequence_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('>'):
            # Remover caracteres no válidos y convertir a mayúsculas
            clean_line = ''.join(c.upper() for c in line if c.upper() in 'ATCGN-')
            sequence_lines.append(clean_line)
    
    return ''.join(sequence_lines)

def test_nvidia_nim_api():
    """Prueba simple de la API de NVIDIA NIM con paciente_juana3.fasta"""
    
    print("=== Test de API NVIDIA NIM ===")
    print(f"API Key disponible: {'Sí' if NIM_KEY else 'No'}")
    
    if not NIM_KEY:
        print("❌ Error: NVIDIA_NIM_KEY no está configurada en el archivo .env")
        return
    
    # Leer el archivo paciente_juana3.fasta
    try:
        with open("paciente_juana3.fasta", "r") as f:
            fasta_content = f.read()
        print("✅ Archivo paciente_juana3.fasta leído correctamente")
        print(f"Contenido (primeros 200 chars): {fasta_content[:200]}...")
    except FileNotFoundError:
        print("❌ Error: No se encontró el archivo paciente_juana3.fasta")
        return
    
    # Extraer secuencia
    sequence = extract_sequence_from_fasta(fasta_content)
    print(f"✅ Secuencia extraída, longitud: {len(sequence)} bases")
    print(f"Primeros 100 chars: {sequence[:100]}")
    
    # Preparar la solicitud a la API
    API_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
    
    headers = {"Authorization": f"Bearer {NIM_KEY}"}
    payload = {
        "sequence": sequence,
        "output_layers": ["blocks.24.inner_mha_cls"]
    }
    
    print(f"\n🔄 Enviando solicitud a: {API_URL}")
    print(f"Longitud de secuencia: {len(sequence)}")
    print("Headers preparados...")
    
    try:
        # Hacer la solicitud
        print("⏳ Enviando solicitud... (esto puede tomar varios minutos)")
        response = requests.post(
            url=API_URL,
            headers=headers,
            json=payload,
            timeout=300  # 5 minutos de timeout
        )
        
        print(f"📡 Respuesta recibida - Status Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'No especificado')}")
        
        if response.status_code != 200:
            print(f"❌ Error HTTP {response.status_code}")
            print(f"Respuesta: {response.text[:500]}...")
            return
        
        # Procesar respuesta según el content-type
        content_type = response.headers.get('content-type', '')
        
        if 'application/zip' in content_type:
            print("🔍 Procesando respuesta ZIP...")
            
            try:
                # Leer el contenido ZIP
                zip_data = io.BytesIO(response.content)
                
                with zipfile.ZipFile(zip_data, 'r') as zip_file:
                    print(f"✅ ZIP abierto correctamente")
                    print(f"Archivos en ZIP: {zip_file.namelist()}")
                    
                    # Buscar archivos relevantes
                    for file_name in zip_file.namelist():
                        print(f"\n📄 Procesando archivo: {file_name}")
                        
                        with zip_file.open(file_name) as file:
                            file_content = file.read()
                            
                        if file_name.endswith('.json'):
                            # Procesar archivo JSON
                            try:
                                json_data = json.loads(file_content.decode('utf-8'))
                                print(f"✅ JSON procesado: {list(json_data.keys())}")
                                
                                if 'data' in json_data:
                                    print("✅ Campo 'data' encontrado en JSON")
                                    
                                    # Intentar decodificar base64
                                    try:
                                        tensor_data = base64.b64decode(json_data['data'])
                                        print(f"✅ Base64 decodificado, tamaño: {len(tensor_data)} bytes")
                                        
                                        # Intentar cargar con numpy
                                        try:
                                            tensor_dict = np.load(io.BytesIO(tensor_data))
                                            print(f"✅ NumPy cargado, claves disponibles: {list(tensor_dict.keys())}")
                                            
                                            if 'blocks.24.inner_mha_cls.output' in tensor_dict:
                                                tensor = tensor_dict['blocks.24.inner_mha_cls.output']
                                                print(f"✅ Tensor extraído exitosamente!")
                                                print(f"Forma del tensor: {tensor.shape}")
                                                print(f"Tipo de datos: {tensor.dtype}")
                                                print(f"Primeros 5 valores: {tensor.flat[:5]}")
                                                
                                                # Intentar guardar como pickle
                                                import pickle
                                                try:
                                                    pickle_data = pickle.dumps(tensor)
                                                    print(f"✅ Pickle creado, tamaño: {len(pickle_data)} bytes")
                                                    
                                                    # Guardar archivo de prueba
                                                    with open("test_embedding.pkl", "wb") as f:
                                                        f.write(pickle_data)
                                                    print("✅ Archivo test_embedding.pkl guardado exitosamente")
                                                    return  # Éxito completo
                                                    
                                                except Exception as e:
                                                    print(f"❌ Error creando pickle: {e}")
                                                    
                                            else:
                                                print(f"❌ Clave 'blocks.24.inner_mha_cls.output' no encontrada")
                                                print(f"Claves disponibles: {list(tensor_dict.keys())}")
                                                
                                        except Exception as e:
                                            print(f"❌ Error cargando con NumPy: {e}")
                                            
                                    except Exception as e:
                                        print(f"❌ Error decodificando base64: {e}")
                                        
                                else:
                                    print(f"❌ Campo 'data' no encontrado en JSON")
                                    print(f"Claves disponibles: {list(json_data.keys())}")
                                    
                            except Exception as e:
                                print(f"❌ Error procesando JSON: {e}")
                                
                        elif file_name.endswith(('.npz', '.npy')):
                            # Procesar archivo numpy directamente
                            try:
                                tensor_dict = np.load(io.BytesIO(file_content))
                                print(f"✅ NumPy directo cargado")
                                
                                if hasattr(tensor_dict, 'keys'):
                                    print(f"Claves disponibles: {list(tensor_dict.keys())}")
                                else:
                                    print(f"Array directo, forma: {tensor_dict.shape}")
                                    
                            except Exception as e:
                                print(f"❌ Error cargando NumPy directo: {e}")
                                
                        else:
                            print(f"Archivo tipo desconocido, tamaño: {len(file_content)} bytes")
                            print(f"Primeros 100 bytes: {file_content[:100]}")
                            
            except Exception as e:
                print(f"❌ Error procesando ZIP: {e}")
                
        elif 'application/json' in content_type:
            print("🔍 Procesando respuesta JSON...")
            try:
                # Código original para JSON
                rj = response.json()
                print("✅ JSON parseado correctamente")
                print(f"Claves en respuesta: {list(rj.keys())}")
                
                if 'data' in rj:
                    print("✅ Campo 'data' encontrado")
                    
                    # Intentar decodificar base64
                    try:
                        tensor_data = base64.b64decode(rj['data'])
                        print(f"✅ Base64 decodificado, tamaño: {len(tensor_data)} bytes")
                        
                        # Intentar cargar con numpy
                        try:
                            tensor_dict = np.load(io.BytesIO(tensor_data))
                            print(f"✅ NumPy cargado, claves disponibles: {list(tensor_dict.keys())}")
                            
                            if 'blocks.24.inner_mha_cls.output' in tensor_dict:
                                tensor = tensor_dict['blocks.24.inner_mha_cls.output']
                                print(f"✅ Tensor extraído exitosamente!")
                                print(f"Forma del tensor: {tensor.shape}")
                                print(f"Tipo de datos: {tensor.dtype}")
                                print(f"Primeros 5 valores: {tensor.flat[:5]}")
                                
                                # Intentar guardar como pickle
                                import pickle
                                try:
                                    pickle_data = pickle.dumps(tensor)
                                    print(f"✅ Pickle creado, tamaño: {len(pickle_data)} bytes")
                                    
                                    # Guardar archivo de prueba
                                    with open("test_embedding.pkl", "wb") as f:
                                        f.write(pickle_data)
                                    print("✅ Archivo test_embedding.pkl guardado exitosamente")
                                    
                                except Exception as e:
                                    print(f"❌ Error creando pickle: {e}")
                                    
                            else:
                                print(f"❌ Clave 'blocks.24.inner_mha_cls.output' no encontrada")
                                
                        except Exception as e:
                            print(f"❌ Error cargando con NumPy: {e}")
                            
                    except Exception as e:
                        print(f"❌ Error decodificando base64: {e}")
                        
                else:
                    print(f"❌ Campo 'data' no encontrado en la respuesta")
                
            except Exception as e:
                print(f"❌ Error parseando JSON: {e}")
                
        else:
            print(f"❌ Content-Type no soportado: {content_type}")
            print(f"Respuesta raw (primeros 200 chars): {response.text[:200]}...")
        
    except requests.exceptions.Timeout:
        print("❌ Timeout - La solicitud tardó más de 5 minutos")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error en la solicitud: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    test_nvidia_nim_api()