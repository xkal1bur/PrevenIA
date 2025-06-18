#!/usr/bin/env python3
"""
Script para dividir cr13.fasta en 1000 partes de referencia y subirlas a S3
"""

import os
import boto3
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración AWS
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Cliente S3
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    region_name=AWS_REGION
)

def extract_sequence_from_fasta(fasta_content):
    """Extrae la secuencia de un archivo FASTA, removiendo headers"""
    lines = fasta_content.split('\n')
    sequence_lines = []
    
    for line in lines:
        line = line.strip()
        if line and not line.startswith('>'):
            # Remover caracteres no válidos y convertir a mayúsculas
            clean_line = ''.join(c.upper() for c in line if c.upper() in 'ATCGN-')
            sequence_lines.append(clean_line)
    
    return ''.join(sequence_lines)

def split_fasta_to_1000_parts():
    """Divide cr13.fasta en 1000 partes de referencia y las sube a S3"""
    
    # 1. Leer el archivo cr13.fasta local
    cr13_file = "cr13.fasta"
    if not os.path.exists(cr13_file):
        print(f"❌ Archivo {cr13_file} no encontrado")
        print("💡 Descargando cr13.fasta desde S3 (raíz del bucket)...")
        try:
            # Intentar descargar desde la raíz del bucket S3
            s3_client.download_file(BUCKET_NAME, "cr13.fasta", cr13_file)
            print("✅ cr13.fasta descargado desde S3")
        except Exception as e:
            print(f"❌ Error descargando cr13.fasta desde S3: {e}")
            return
    
    print(f"📁 Leyendo archivo {cr13_file}...")
    with open(cr13_file, 'r') as f:
        fasta_content = f.read()
    
    # 2. Extraer la secuencia sin headers
    sequence = extract_sequence_from_fasta(fasta_content)
    sequence_length = len(sequence)
    print(f"📏 Longitud de la secuencia: {sequence_length:,} bases")
    
    # 3. Calcular tamaño de cada parte (1000 partes de referencia)
    chunk_size = sequence_length // 1000
    remainder = sequence_length % 1000
    
    print(f"🔢 Dividiendo en 1000 partes de referencia de ~{chunk_size:,} bases cada una")
    
    # 4. Verificar si cr13.fasta ya existe en la raíz del bucket
    print("📤 Verificando cr13.fasta en la raíz del bucket...")
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key="cr13.fasta")
        print("✅ cr13.fasta ya existe en la raíz del bucket")
    except:
        print("📤 Subiendo cr13.fasta a la raíz del bucket...")
        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key="cr13.fasta",
                Body=fasta_content.encode('utf-8'),
                ContentType="text/plain"
            )
            print("✅ cr13.fasta subido a la raíz del bucket")
        except Exception as e:
            print(f"❌ Error subiendo cr13.fasta: {e}")
            return
    
    # 5. Dividir y subir cada parte de referencia a split_fasta_files/
    print("🔄 Dividiendo y subiendo 1000 fragmentos de referencia...")
    
    success_count = 0
    for i in range(1000):
        # Calcular posiciones de inicio y fin
        start_pos = i * chunk_size
        if i < remainder:
            start_pos += i
            end_pos = start_pos + chunk_size + 1
        else:
            start_pos += remainder
            end_pos = start_pos + chunk_size
        
        # Extraer el fragmento
        chunk_sequence = sequence[start_pos:end_pos]
        
        # Crear contenido FASTA para este fragmento de referencia
        chunk_filename = f"default_part_{i+1:04d}.fasta"  # 0001, 0002, ..., 1000
        chunk_fasta_content = f">chr13_reference_part_{i+1:04d} | positions {start_pos+1}-{end_pos} | length {len(chunk_sequence)}\n{chunk_sequence}\n"
        
        # Subir a S3 en split_fasta_files/
        try:
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=f"split_fasta_files/{chunk_filename}",
                Body=chunk_fasta_content.encode('utf-8'),
                ContentType="text/plain"
            )
            success_count += 1
            
            if (i + 1) % 100 == 0:  # Mostrar progreso cada 100 archivos
                print(f"📤 Subidos {i+1}/1000 fragmentos de referencia... ({success_count} exitosos)")
                
        except Exception as e:
            print(f"❌ Error subiendo {chunk_filename}: {e}")
            continue
    
    print("✅ ¡Proceso completado!")
    print(f"📊 Resumen:")
    print(f"   - Secuencia original: {sequence_length:,} bases")
    print(f"   - Fragmentos de referencia creados: {success_count}/1000")
    print(f"   - Tamaño promedio por fragmento: {chunk_size:,} bases")
    print(f"   - Archivos subidos a: split_fasta_files/default_part_0001.fasta - default_part_1000.fasta")
    print(f"   - Archivo completo en: cr13.fasta (raíz del bucket)")

def list_uploaded_files():
    """Lista los archivos subidos para verificación"""
    print("\n🔍 Verificando archivos de referencia subidos...")
    
    try:
        # Verificar cr13.fasta en la raíz del bucket
        s3_client.head_object(Bucket=BUCKET_NAME, Key="cr13.fasta")
        print("✅ cr13.fasta encontrado en la raíz del bucket")
    except:
        print("❌ cr13.fasta NO encontrado en la raíz del bucket")
    
    # Listar archivos en split_fasta_files/
    try:
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix="split_fasta_files/default_part_"
        )
        
        files = response.get('Contents', [])
        print(f"📁 Fragmentos de referencia en split_fasta_files/: {len(files)}")
        
        if len(files) > 0:
            # Ordenar por nombre para obtener el orden correcto
            files.sort(key=lambda x: x['Key'])
            print(f"   Primer fragmento: {files[0]['Key']}")
            print(f"   Último fragmento: {files[-1]['Key']}")
            
            # Verificar que tenemos exactamente 1000 fragmentos
            expected_files = set(f"split_fasta_files/default_part_{i:04d}.fasta" for i in range(1, 1001))
            actual_files = set(f['Key'] for f in files if f['Key'].startswith('split_fasta_files/default_part_'))
            
            missing_files = expected_files - actual_files
            extra_files = actual_files - expected_files
            
            if missing_files:
                print(f"⚠️  Archivos faltantes: {len(missing_files)} (ej: {list(missing_files)[:5]})")
            if extra_files:
                print(f"⚠️  Archivos extra: {len(extra_files)} (ej: {list(extra_files)[:5]})")
            
            if not missing_files and not extra_files:
                print("✅ Todos los 1000 fragmentos de referencia están presentes")
            
    except Exception as e:
        print(f"❌ Error listando archivos: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando división de cr13.fasta en 1000 partes de referencia...")
    split_fasta_to_1000_parts()
    list_uploaded_files()
    print("\n🎉 Script completado. Ahora tienes 1000 fragmentos de referencia disponibles.") 