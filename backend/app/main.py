import os
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import boto3
from datetime import datetime, timedelta
import models, schemas, crud, auth
from database import SessionLocal, engine
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from ml_inference import ml_service
from fastapi.responses import StreamingResponse
from urllib.parse import unquote
import re
import requests
import numpy as np
import base64
import io
import pickle
import zipfile
import json
from sqlalchemy import extract, func, distinct, text, inspect
from fastapi import Response
import asyncio
import concurrent.futures
from functools import partial
import time

models.Base.metadata.create_all(bind=engine)

# -----------------------------------------------------------------
# Garantizar que la columna "created_at" exista en la tabla paciente
# (puede faltar si la base fue creada antes de añadir la columna al modelo)
# -----------------------------------------------------------------
def _ensure_paciente_created_at_column():
    """Comprueba si la columna paciente.created_at existe; si no, la crea."""
    with engine.connect() as conn:
        inspector = inspect(conn)
        columns = [col["name"] for col in inspector.get_columns("paciente")]
        if "created_at" not in columns:
            # Agregar la columna con valor por defecto NOW()
            conn.execute(text("ALTER TABLE paciente ADD COLUMN created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();"))
            # Asignar fecha actual a registros antiguos
            conn.execute(text("UPDATE paciente SET created_at = NOW() WHERE created_at IS NULL;"))
            conn.commit()
            print("✅ Columna paciente.created_at añadida y valores inicializados")

# Ejecutar inmediatamente al importar el módulo
try:
    _ensure_paciente_created_at_column()
except Exception as e:
    # Solo informar; no queremos impedir que la app arranque
    print(f"⚠️  No se pudo verificar/crear paciente.created_at: {e}")

app = FastAPI(title="API del Prevenia")

# Evento de startup para precargar la secuencia de referencia y crear DB BLAST
@app.on_event("startup")
async def startup_event():
    """Precarga la secuencia cr13 y crea la base de datos BLAST al iniciar la aplicación"""
    try:
        # Precargar ambos caches
        get_cr13_reference()
        get_cr13_sequence()
        print("✅ Secuencia cr13.fasta precargada en memoria (FASTA + extraída)")
        
        # Crear base de datos BLAST
        await create_blast_database()
        print("✅ Base de datos BLAST cr13 creada")
    except Exception as e:
        print(f"⚠️ Error en startup: {e}")

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/pacientes", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN     = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION            = os.getenv("AWS_REGION", "us-east-1")
NIM_KEY               = os.getenv("NVIDIA_NIM_KEY")
BUCKET_NAME           = os.getenv("S3_BUCKET_NAME")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,     
    region_name=AWS_REGION
)


# Cache para la secuencia de referencia del cromosoma 13
_cr13_reference_cache = None
_cr13_sequence_cache = None  # Cache para la secuencia extraída (sin headers)

def get_cr13_reference():
    """Obtiene la secuencia de referencia cr13, usando caché si está disponible"""
    global _cr13_reference_cache
    
    if _cr13_reference_cache is not None:
        return _cr13_reference_cache
    
    try:
        # Intentar cargar desde archivo local primero
        try:
            with open("cr13.fasta", "r") as f:
                _cr13_reference_cache = f.read()
                return _cr13_reference_cache
        except FileNotFoundError:
            pass
        
        # Si no está local, descargar desde S3 (raíz del bucket) y cachear
        ref_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key="cr13.fasta")
        _cr13_reference_cache = ref_obj["Body"].read().decode('utf-8')
        
        # Opcionalmente guardar local para próximas veces
        try:
            with open("cr13.fasta", "w") as f:
                f.write(_cr13_reference_cache)
        except:
            pass  # No es crítico si no se puede guardar
        
        return _cr13_reference_cache
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo obtener la secuencia de referencia cr13.fasta: {e}")

def get_cr13_sequence():
    """Obtiene la secuencia de referencia cr13 extraída (sin headers), usando caché"""
    global _cr13_sequence_cache
    
    if _cr13_sequence_cache is not None:
        return _cr13_sequence_cache
    
    # Obtener el contenido FASTA y extraer la secuencia
    ref_content = get_cr13_reference()
    _cr13_sequence_cache = extract_sequence_from_fasta(ref_content)
    
    return _cr13_sequence_cache

async def create_blast_database():
    """Crea la base de datos BLAST para cr13.fasta"""
    # Crear directorio para la base de datos si no existe
    os.makedirs("blast_db", exist_ok=True)
    
    # Guardar la secuencia de referencia localmente si no existe
    if not os.path.exists("cr13.fasta"):
        ref_content = get_cr13_reference()
        with open("cr13.fasta", "w") as f:
            f.write(ref_content)
    
    # Crear la base de datos BLAST
    try:
        result = subprocess.run([
            'makeblastdb',
            '-in', 'cr13.fasta',
            '-dbtype', 'nucl',
            '-out', 'blast_db/chr13_db'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise Exception(f"Error creando base de datos BLAST: {result.stderr}")
            
    except FileNotFoundError:
        raise Exception("makeblastdb no encontrado. Instale BLAST+ toolkit")
    except subprocess.TimeoutExpired:
        raise Exception("Timeout creando base de datos BLAST")


### ——— RUTAS PARA DOCTORES ——— ###

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Bienvenido a la API del Prevenia"}


@app.post("/register/doctor", response_model=schemas.DoctorOut, tags=["Doctors"])
def register_doctor(doctor: schemas.DoctorCreate, db: Session = Depends(get_db)):
    if crud.get_doctor_by_email(db, doctor.correo):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ya existe un doctor con ese correo"
        )
    nuevo = crud.create_doctor(db, doctor)
    return nuevo


@app.post("/login/doctor", response_model=schemas.Token, tags=["Doctors"])
def login_doctor(form_data: schemas.DoctorLogin, db: Session = Depends(get_db)):
    db_doc = crud.get_doctor_by_email(db, form_data.correo)
    if not db_doc or not auth.verify_password(form_data.password, db_doc.hashed_pw):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth.create_access_token({"sub": db_doc.correo})
    return {"access_token": token, "token_type": "bearer"}


@app.get("/doctors/me", response_model=schemas.DoctorProfile, tags=["Doctors"])
def read_current_doctor(current_doc: models.Doctor = Depends(auth.get_current_doctor)):
    return current_doc


### ——— RUTAS PARA PACIENTES ——— ###

@app.post(
    "/register/paciente",
    response_model=schemas.PacienteOut,
    status_code=status.HTTP_201_CREATED,
    tags=["Pacientes"]
)
def register_paciente(
    dni: str         = Form(..., description="DNI del paciente"),
    nombres: str     = Form(..., description="Nombres del paciente"),
    apellidos: str   = Form(..., description="Apellidos del paciente"),
    edad: int        = Form(..., description="Edad del paciente"),
    celular: str     = Form(..., description="Número de celular del paciente"),
    correo: str      = Form(..., description="Correo del paciente"),
    foto: UploadFile = File(..., description="Foto del paciente (JPEG/PNG)"),
    db: Session      = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    if crud.get_paciente_por_dni(db, dni):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ya existe un paciente con ese DNI"
        )
    if crud.get_paciente_por_correo(db, correo):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ya existe un paciente con ese correo"
        )
    paciente_data = schemas.PacienteCreate(
        dni=dni,
        nombres=nombres,
        apellidos=apellidos,
        edad=edad,
        celular=celular,
        correo=correo,
        doctor_id=current_doc.id
    )

    nuevo_paciente = crud.create_paciente(db, paciente_data, foto)
    return nuevo_paciente


@app.get(
    "/pacientes/{paciente_id}",
    response_model=schemas.PacienteOut,
    tags=["Pacientes"]
)
def get_paciente(paciente_id: int, db: Session = Depends(get_db)):
    paciente = crud.get_paciente_por_id(db, paciente_id)
    if not paciente:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paciente no encontrado"
        )
    return paciente


@app.get(
    "/pacientes",
    response_model=list[schemas.PacienteOut],
    tags=["Pacientes"]
)
def list_pacientes(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    pacientes = crud.get_pacientes(db, skip=skip, limit=limit)
    return pacientes


@app.get(
    "/pacientes/doctor/{doctor_id}",
    response_model=list[schemas.PacienteOut],
    tags=["Pacientes"]
)
def list_pacientes_por_doctor(
    doctor_id: int,
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    
    if current_doc.id != doctor_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tienes permiso para ver pacientes de otro doctor."
        )
    return crud.get_pacientes_por_doctor(db, doctor_id)

@app.get("/pacientes/dni/{dni}",response_model=schemas.PacienteOut,tags=["Pacientes"])
def get_paciente_por_dni(dni: str, db: Session = Depends(get_db), current_doc: models.Doctor = Depends(auth.get_current_doctor)):
    paciente = crud.get_paciente_por_dni(db, dni)
    if not paciente or paciente.doctor_id != current_doc.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paciente no encontrado o no te pertenece"
        )
    return paciente


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

def extract_sequences_from_multi_fasta(fasta_content: str) -> list[str]:
    """Devuelve una lista con cada secuencia presente en un FASTA multi-entrada."""
    sequences: list[str] = []
    current: list[str] = []

    for line in fasta_content.split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('>'):
            # Guardar la secuencia previa (si existe)
            if current:
                sequences.append(''.join(current))
                current = []
            continue  # Omitir la línea del header

        # Limpiar la línea y añadirla a la secuencia actual
        clean_line = ''.join(c.upper() for c in line if c.upper() in 'ATCGN-')
        current.append(clean_line)

    # Añadir la última secuencia si quedó algo
    if current:
        sequences.append(''.join(current))

    return sequences

def extract_mismatches_from_fasta(fasta_content: str) -> list[tuple[int, str]]:
    """
    Extrae información de mismatches de un archivo FASTA con headers que contienen posición.
    Retorna una lista de tuplas (posición, secuencia_alterada).
    """
    mismatches = []
    current_sequence = ""
    current_position = None
    
    for line in fasta_content.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            # Si hay una secuencia acumulada, agregarla a la lista
            if current_sequence and current_position is not None:
                clean_seq = ''.join(c for c in current_sequence.upper() if c in 'ATCGN-')
                if clean_seq:
                    mismatches.append((current_position, clean_seq))
                current_sequence = ""
                current_position = None
            
            # Extraer posición del header (formato: >Mismatch_1_pos_12345_window_...)
            try:
                parts = line.split('_')
                # Buscar la parte que dice 'pos' y tomar el siguiente número
                for i, part in enumerate(parts):
                    if part == 'pos' and i + 1 < len(parts):
                        current_position = int(parts[i + 1])
                        break
            except (ValueError, IndexError):
                print(f"⚠️ [EMBEDDING] No se pudo extraer posición del header: {line}")
                
        else:
            # Acumular la secuencia
            current_sequence += line
    
    # Agregar la última secuencia si existe
    if current_sequence and current_position is not None:
        clean_seq = ''.join(c for c in current_sequence.upper() if c in 'ATCGN-')
        if clean_seq:
            mismatches.append((current_position, clean_seq))
    
    return mismatches

# Constantes para parse_sequences3
WINDOW_SIZE = 128
complement_table = str.maketrans("ATCG", "TAGC")

def parse_sequences3(mid_pos: int, alt_seq: str, chr13_seq: str) -> tuple[str, str, str, str]:
    """
    Genera 4 secuencias para una mutación dada:
    - ref_seq: secuencia de referencia 
    - var_seq: secuencia variante
    - ref_seq_rev_comp: complemento reverso de referencia
    - var_seq_rev_comp: complemento reverso de variante
    """
    p = mid_pos - 1  # Convertir a índice base-0
    full_seq = chr13_seq

    ref_seq_start = p - WINDOW_SIZE//2
    ref_seq_end = p + WINDOW_SIZE//2
    ref_seq = full_seq[ref_seq_start:ref_seq_end+1]

    dash_count = alt_seq.count("-")
    var_seq = alt_seq.replace("-", "")
    var_seq += full_seq[ref_seq_end+1:ref_seq_end+1+dash_count]

    ref_seq_rev_comp = ref_seq.translate(complement_table)[::-1]
    var_seq_rev_comp = var_seq.translate(complement_table)[::-1]

    # Verificar que todas las secuencias tienen la longitud correcta
    assert len(var_seq) == len(ref_seq) == WINDOW_SIZE + 1, f"Longitudes incorrectas: ref={len(ref_seq)}, var={len(var_seq)}, esperado={WINDOW_SIZE + 1}"

    return ref_seq, var_seq, ref_seq_rev_comp, var_seq_rev_comp

import subprocess
import tempfile

@app.post("/pacientes/{dni}/upload_fasta")
async def upload_fasta(
    dni: str = Path(..., description="DNI del paciente"),
    fasta_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    
    if not fasta_file.filename.lower().endswith((".fasta", ".fa", ".fna")):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .fasta/.fa/.fna")

    try:
        # Leer el contenido del archivo
        contents = await fasta_file.read()

        # Subir archivo directamente a S3 sin procesamiento
        original_key = f"{dni}/{fasta_file.filename}"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=original_key,
            Body=contents,
            ContentType="application/octet-stream"
        )

        return JSONResponse(
            status_code=200, 
            content={
                "message": f"Archivo {fasta_file.filename} subido correctamente",
                "key": original_key,
                "filename": fasta_file.filename
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error subiendo archivo: {e}")

@app.post("/pacientes/{dni}/upload_file")
async def upload_file(
    dni: str = Path(..., description="DNI del paciente"),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint para subir archivos generales (PKL, etc.) sin procesamiento específico
    """
    
    # Validar extensiones permitidas
    allowed_extensions = [".pkl", ".json", ".txt", ".csv"]
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Solo se permiten archivos con extensiones: {', '.join(allowed_extensions)}"
        )

    try:
        # Leer el contenido del archivo
        contents = await file.read()

        # Subir archivo directamente a S3 sin procesamiento
        file_key = f"{dni}/{file.filename}"
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=file_key,
            Body=contents,
            ContentType="application/octet-stream"
        )

        return JSONResponse(
            status_code=200, 
            content={
                "message": f"Archivo {file.filename} subido correctamente",
                "key": file_key,
                "filename": file.filename,
                "type": "general_file",
                "extension": file_ext
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error subiendo archivo: {e}")

async def upload_chunk_to_s3(s3_client, bucket_name: str, key: str, body: str):
    """Función auxiliar para subir un chunk a S3 de forma asíncrona"""
    loop = asyncio.get_event_loop()
    
    # Ejecutar la operación S3 en un thread pool para no bloquear el event loop
    with concurrent.futures.ThreadPoolExecutor() as executor:
        await loop.run_in_executor(
            executor,
            partial(
                s3_client.put_object,
                Bucket=bucket_name,
                Key=key,
                Body=body.encode('utf-8'),
                ContentType="text/plain"
            )
        )

async def upload_chunks_parallel(s3_client, bucket_name: str, patient_folder: str, sequence: str, chunk_count: int = 1000):
    """Sube chunks a S3 en paralelo para acelerar el proceso"""
    
    # Dividir la secuencia en chunks
    sequence_length = len(sequence)
    chunk_size = sequence_length // chunk_count
    remainder = sequence_length % chunk_count
    
    # Crear todas las tareas de subida
    upload_tasks = []
    
    for i in range(chunk_count):
        start_pos = i * chunk_size
        # Distribuir el remainder entre las primeras partes
        if i < remainder:
            start_pos += i
            end_pos = start_pos + chunk_size + 1
        else:
            start_pos += remainder
            end_pos = start_pos + chunk_size
        
        chunk_sequence = sequence[start_pos:end_pos]
        chunk_filename = f"patient_part_{i+1:04d}.fasta"  # 0001, 0002, ..., 1000
        chunk_key = f"{patient_folder}{chunk_filename}"
        
        # Crear tarea asíncrona para subir este chunk
        task = upload_chunk_to_s3(s3_client, bucket_name, chunk_key, chunk_sequence)
        upload_tasks.append(task)
    
    # Ejecutar todas las subidas en paralelo con un límite de concurrencia
    # Procesar en lotes para no sobrecargar S3
    batch_size = 50  # Subir máximo 50 archivos concurrentemente
    
    for i in range(0, len(upload_tasks), batch_size):
        batch = upload_tasks[i:i + batch_size]
        await asyncio.gather(*batch)
        
        # Pequeña pausa entre lotes para ser amigable con S3
        if i + batch_size < len(upload_tasks):
            await asyncio.sleep(0.1)

async def process_patient_file_with_blast(dni: str, filename: str, contents: bytes, fasta_text: str):
    """Procesa archivo de paciente con BLAST y lo divide en 1000 partes"""
    
    # Extraer secuencia del FASTA
    sequences = extract_sequences_from_multi_fasta(fasta_text)
    if not sequences:
        raise HTTPException(status_code=400, detail="No se encontraron secuencias válidas en el archivo FASTA")
    
    # Usar la primera secuencia encontrada
    query_seq = sequences[0].replace('\n', '').replace('\r', '').upper()
    query_seq_clean = ''.join(c for c in query_seq if c in 'ATCGN-')
    
    # Crear archivo temporal para BLAST
    import tempfile
    import uuid
    import subprocess
    
    query_temp_path = f"/tmp/query_{uuid.uuid4().hex}.fasta"
    results_temp_path = f"/tmp/blast_results_{uuid.uuid4().hex}.tsv"
    
    try:
        # Escribir secuencia query en formato FASTA
        with open(query_temp_path, 'w') as f:
            f.write(f">query\n{query_seq_clean}\n")
        
        # Ejecutar BLAST
        blast_cmd = [
            "blastn",
            "-query", query_temp_path,
            "-db", "app/blast_db/chr13_db",
            "-out", results_temp_path,
            "-outfmt", "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore",
            "-max_target_seqs", "1",
            "-evalue", "1e-5"
        ]
        
        result = subprocess.run(blast_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error ejecutando BLAST: {result.stderr}")
        
        # Leer resultados de BLAST
        with open(results_temp_path, 'r') as f:
            blast_results = f.read().strip()
        
        if not blast_results:
            raise HTTPException(status_code=400, detail="No se encontraron alineamientos significativos con BLAST")
        
        # Parsear la primera línea de resultados (mejor hit)
        blast_fields = blast_results.split('\n')[0].split('\t')
        qseqid, sseqid, pident, length, mismatch, gapopen, qstart, qend, sstart, send, evalue, bitscore = blast_fields
        
        # Obtener la secuencia de referencia cr13
        cr13_seq = get_cr13_sequence()
        if not cr13_seq:
            raise HTTPException(status_code=500, detail="No se pudo cargar la secuencia de referencia cr13")
        
        # Calcular posición y longitud del alineamiento
        ref_start = int(sstart) - 1  # Convertir a índice base-0
        ref_end = int(send)
        aligned_length = len(cr13_seq)
        
        # Crear secuencia alineada de forma más eficiente
        patient_aligned_str = (
            '-' * ref_start + 
            query_seq_clean + 
            '-' * max(0, aligned_length - ref_start - len(query_seq_clean))
        )
        
        # Crear carpeta para las partes del paciente
        patient_folder = f"{dni}/patient_chunks/"
        
        # Subir chunks en paralelo - OPTIMIZACIÓN PRINCIPAL
        await upload_chunks_parallel(s3_client, BUCKET_NAME, patient_folder, patient_aligned_str)
        
        # Subir archivos adicionales en paralelo también
        additional_uploads = []
        
        # Archivo original
        original_key = f"{dni}/{filename}"
        original_task = upload_chunk_to_s3(s3_client, BUCKET_NAME, original_key, contents.decode('utf-8'))
        additional_uploads.append(original_task)
        
        # Secuencia alineada completa
        aligned_key = f"{dni}/aligned_{filename}"
        aligned_task = upload_chunk_to_s3(s3_client, BUCKET_NAME, aligned_key, patient_aligned_str)
        additional_uploads.append(aligned_task)
        
        # Resultados de BLAST
        blast_results_key = f"{dni}/blast_results_{filename}.tsv"
        blast_task = upload_chunk_to_s3(s3_client, BUCKET_NAME, blast_results_key, blast_results)
        additional_uploads.append(blast_task)
        
        # Ejecutar subidas adicionales en paralelo
        await asyncio.gather(*additional_uploads)
        
        return JSONResponse(
            status_code=200, 
            content={
                "message": f"Archivo {filename} procesado con BLAST contra cr13.fasta y dividido en 1000 partes con éxito",
                "original_key": original_key,
                "aligned_key": aligned_key,
                "blast_results_key": blast_results_key,
                "chunks_folder": patient_folder,
                "sequence_length": len(patient_aligned_str),
                "chunks_created": 1000,
                "blast_stats": {
                    "percent_identity": float(pident),
                    "alignment_length": int(length),
                    "ref_start": int(sstart),
                    "ref_end": int(send),
                    "evalue": float(evalue)
                },
                "alignment_method": "blast",
                "reference_used": "blast_db/chr13_db"
            }
        )
        
    finally:
        # Limpiar archivos temporales
        import os
        try:
            os.unlink(query_temp_path)
            os.unlink(results_temp_path)
        except:
            pass

async def process_normal_fasta(dni: str, filename: str, contents: bytes, fasta_text: str):
    """Procesa un archivo FASTA normal dividiéndolo en 1000 partes iguales"""
    
    # Extraer la secuencia sin headers
    sequence = extract_sequence_from_fasta(fasta_text)
    
    if not sequence:
        raise HTTPException(status_code=400, detail="No se encontró secuencia válida en el archivo FASTA")
    
    # Crear carpeta para las partes del paciente
    patient_folder = f"{dni}/patient_chunks/"
    
    # Subir chunks en paralelo - OPTIMIZACIÓN PRINCIPAL
    await upload_chunks_parallel(s3_client, BUCKET_NAME, patient_folder, sequence)
    
    # Subir archivo original
    original_key = f"{dni}/{filename}"
    await upload_chunk_to_s3(s3_client, BUCKET_NAME, original_key, contents.decode('utf-8'))
    
    return JSONResponse(
        status_code=200, 
        content={
            "message": "Archivo procesado y dividido en 1000 partes con éxito",
            "original_key": original_key,
            "chunks_folder": patient_folder,
            "sequence_length": len(sequence),
            "chunks_created": 1000
        }
    )


### ——— ENDPOINT PARA PREDICCIONES DE ML ——— ###

@app.get("/predictions/{dni}", response_model=schemas.PredictionsResponse, tags=["ML Predictions"])
def get_predictions_for_patient_get(
    dni: str = Path(..., description="DNI del paciente"),
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """Obtener predicciones de ML para un paciente específico (método GET - usa embedding por defecto)"""
    try:
        # Verificar que el paciente existe y pertenece al doctor
        paciente = crud.get_paciente_por_dni(db, dni)
        if not paciente or paciente.doctor_id != current_doc.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Paciente no encontrado o no te pertenece"
            )
        
        # Obtener predicciones personalizadas para el paciente (usa embedding por defecto)
        return ml_service.get_predictions_for_patient(dni, paciente.nombres, paciente.apellidos)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo predicciones: {str(e)}")

@app.post("/predictions/{dni}", response_model=schemas.PredictionsResponse, tags=["ML Predictions"])
def get_predictions_with_embedding(
    dni: str = Path(..., description="DNI del paciente"),
    embedding_filename: str = Form(..., description="Nombre del archivo .pkl con el embedding"),
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """Obtener predicciones de ML usando un archivo de embedding específico (.pkl)"""
    
    print(f"🔬 [PREDICTIONS] Iniciando predicciones con embedding específico")
    print(f"   📋 DNI: {dni}")
    print(f"   📄 Archivo embedding: {embedding_filename}")
    print(f"   👨‍⚕️ Doctor: {current_doc.nombre} (ID: {current_doc.id})")
    
    try:
        # Verificar que el paciente existe y pertenece al doctor
        print(f"🔍 [PREDICTIONS] Verificando paciente...")
        paciente = crud.get_paciente_por_dni(db, dni)
        if not paciente or paciente.doctor_id != current_doc.id:
            print(f"❌ [PREDICTIONS] Paciente no encontrado o no autorizado")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Paciente no encontrado o no te pertenece"
            )
        print(f"✅ [PREDICTIONS] Paciente verificado: {paciente.nombres} {paciente.apellidos}")
        
        # Verificar que el archivo de embedding existe en S3
        embedding_key = f"{dni}/{embedding_filename}"
        print(f"📂 [PREDICTIONS] Verificando embedding en S3: {embedding_key}")
        
        try:
            # Verificar que el archivo existe y es un .pkl
            if not embedding_filename.endswith('.pkl'):
                raise HTTPException(status_code=400, detail="El archivo debe ser un embedding .pkl")
            
            embedding_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=embedding_key)
            embedding_data = embedding_obj["Body"].read()
            embedding_size = len(embedding_data)
            print(f"✅ [PREDICTIONS] Embedding encontrado: {embedding_size:,} bytes ({embedding_size/(1024*1024):.2f} MB)")
            
        except s3_client.exceptions.NoSuchKey:
            print(f"❌ [PREDICTIONS] Archivo de embedding no encontrado: {embedding_key}")
            raise HTTPException(status_code=404, detail=f"Archivo de embedding no encontrado: {embedding_filename}")
        except Exception as e:
            print(f"❌ [PREDICTIONS] Error accediendo al embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Error accediendo al archivo de embedding: {e}")
        
        # Deserializar el embedding
        print(f"🔄 [PREDICTIONS] Deserializando embedding...")
        try:
            embedding = pickle.loads(embedding_data)
            print(f"✅ [PREDICTIONS] Embedding deserializado: shape {embedding.shape}, dtype {embedding.dtype}")
            
            # Verificar que el embedding tiene la forma correcta
            if embedding.shape[0] != 32768:
                print(f"⚠️ [PREDICTIONS] Forma de embedding inesperada: {embedding.shape}, esperado (32768,)")
                raise HTTPException(status_code=400, detail=f"Embedding inválido: esperado 32768 características, encontrado {embedding.shape[0]}")
                
        except pickle.UnpicklingError as e:
            print(f"❌ [PREDICTIONS] Error deserializando pickle: {e}")
            raise HTTPException(status_code=400, detail="Archivo de embedding corrupto o inválido")
        except Exception as e:
            print(f"❌ [PREDICTIONS] Error procesando embedding: {e}")
            raise HTTPException(status_code=500, detail=f"Error procesando embedding: {e}")
        
        # Obtener predicciones usando el embedding específico
        print(f"🤖 [PREDICTIONS] Generando predicciones con ML...")
        try:
            # Pasar el embedding al servicio ML
            predictions = ml_service.get_predictions_for_patient_with_embedding(
                dni, 
                paciente.nombres, 
                paciente.apellidos, 
                embedding,
                embedding_filename
            )
            
            print(f"✅ [PREDICTIONS] Predicciones generadas exitosamente")
            print(f"   📊 Modelos utilizados: {predictions.get('total_models', 'N/A')}")
            print(f"   🎯 Estado: {predictions.get('status', 'N/A')}")
            
            return predictions
            
        except Exception as e:
            print(f"❌ [PREDICTIONS] Error en servicio ML: {e}")
            raise HTTPException(status_code=500, detail=f"Error generando predicciones: {e}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"💥 [PREDICTIONS] Error inesperado: {e}")
        import traceback
        print(f"📋 [PREDICTIONS] Traceback completo:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")


# LISTAR archivos en S3 para un paciente
@app.get("/pacientes/{dni}/files", tags=["Pacientes"])
def list_fasta_files(
    dni: str,
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """
    Lista los archivos y carpetas de primer nivel para un paciente en S3.
    No lista de forma recursiva, sino que agrupa los chunks en carpetas.
    """
    prefix = f"{dni}/"
    files = []

    try:
        # Usar Delimiter='/' para tratar S3 como un sistema de archivos
        resp = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME, 
            Prefix=prefix, 
            Delimiter='/'
        )

        # 1. Procesar los archivos de primer nivel (devueltos en 'Contents')
        for obj in resp.get("Contents", []):
            # Ignorar el objeto que representa la propia carpeta
            if obj["Key"] == prefix:
                continue

            # El nombre del archivo es la clave sin el prefijo
            filename_without_prefix = obj["Key"].replace(prefix, "", 1)
            
            # Solo añadir si no es una cadena vacía (en caso de algún objeto extraño)
            if filename_without_prefix:
                files.append({
                    "filename": filename_without_prefix,
                    "key": obj["Key"],
                    "lastModified": obj["LastModified"].isoformat(),
                    "isFolder": False, # Es un archivo
                    "size": obj["Size"]
                })

        # 2. Procesar las carpetas (devueltas en 'CommonPrefixes')
        for common_prefix in resp.get("CommonPrefixes", []):
            folder_key = common_prefix.get("Prefix")
            folder_name = folder_key.replace(prefix, "", 1)

            # Determinar el tipo de carpeta y contar los chunks si es necesario
            chunk_count = 0
            folder_type = "generic_folder"
            
            # Para las carpetas de chunks, hacemos una consulta adicional para contarlos
            # y obtener la fecha de modificación más reciente.
            if "patient_chunks/" in folder_name or "aligned_chunks/" in folder_name:
                chunks_resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_key)
                chunk_items = chunks_resp.get("Contents", [])
                chunk_count = len(chunk_items)
                
                # Encontrar la fecha más reciente dentro de la carpeta
                latest_mod_date = max([item["LastModified"] for item in chunk_items]) if chunk_items else None

                files.append({
                    "filename": folder_name,
                    "key": folder_key,
                    "lastModified": latest_mod_date.isoformat() if latest_mod_date else "N/A",
                    "isFolder": True,
                    "chunkCount": chunk_count,
                    "type": "aligned_sequences" if "aligned" in folder_name else "raw_sequences"
                })

    except Exception as e:
        # Es buena práctica manejar errores de la API de S3
        raise HTTPException(status_code=500, detail=f"Error al listar archivos de S3: {e}")

    # Ordenar la lista para una mejor visualización (carpetas primero, luego archivos)
    files.sort(key=lambda x: (not x["isFolder"], x["filename"]))

    return {"files": files}



@app.get("/pacientes/{dni}/patient_chunks/info", tags=["Pacientes"])
def get_patient_chunks_info(
    dni: str,
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """Obtiene información sobre los chunks del paciente y la posición de match"""
    try:
        # Verificar si existen los chunks del paciente
        prefix = f"{dni}/patient_chunks/"
        resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        items = resp.get("Contents", [])
        
        if not items:
            raise HTTPException(status_code=404, detail="No se encontraron chunks del paciente")
        
        # Obtener el primer chunk para encontrar la posición de match
        try:
            first_chunk_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=f"{prefix}patient_part_0001.fasta")
            first_chunk_content = first_chunk_obj["Body"].read().decode('utf-8').strip()
            
            # Encontrar la primera posición que no es "-"
            first_match_position = 0
            for i, char in enumerate(first_chunk_content):
                if char != '-':
                    first_match_position = i
                    break
            
            # Calcular la longitud total estimada
            chunk_length = len(first_chunk_content)
            total_length = chunk_length * 1000  # 1000 chunks del paciente
            
            return {
                "chunks_available": True,
                "chunk_count": len(items),
                "first_match_position": first_match_position,
                "chunk_length": chunk_length,
                "total_length": total_length,
                "chunks_folder": prefix,
                "chunk_type": "patient_chunks"
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error leyendo chunks: {e}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo información de chunks: {e}")

@app.get("/pacientes/{dni}/aligned_chunks/info", tags=["Pacientes"])
def get_aligned_chunks_info(
    dni: str,
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """Obtiene información sobre los chunks alineados del paciente y la posición de match"""
    try:
        # Verificar si existen los chunks alineados
        prefix = f"{dni}/aligned_chunks/"
        resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        items = resp.get("Contents", [])
        
        if not items:
            raise HTTPException(status_code=404, detail="No se encontraron chunks alineados del paciente")
        
        # NO usar el primer chunk - usar información de BLAST directamente
        # Obtener información de longitud de cualquier chunk disponible
        try:
            sample_chunk_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=f"{prefix}aligned_part_0001.fasta")
            sample_chunk_content = sample_chunk_obj["Body"].read().decode('utf-8').strip()
            chunk_length = len(sample_chunk_content)
            total_length = chunk_length * 1000  # 1000 chunks alineados
        except Exception:
            # Fallback si no se puede leer el chunk
            chunk_length = 5000  # Estimación
            total_length = 5000000
            
        # Buscar y leer los resultados de BLAST para obtener información precisa
        blast_results_key = None
        blast_start_position = 0  # Inicializar con 0
        try:
            blast_prefix = f"{dni}/blast_results_"
            blast_resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=blast_prefix)
            blast_items = blast_resp.get("Contents", [])
            if blast_items:
                # Obtener el archivo de resultados BLAST más reciente
                blast_results_key = sorted(blast_items, key=lambda x: x["LastModified"])[-1]["Key"]
                
                # Leer el contenido del archivo BLAST para obtener la posición real
                blast_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=blast_results_key)
                blast_content = blast_obj["Body"].read().decode('utf-8').strip()
                
                if blast_content:
                    # Parsear la primera línea de resultados BLAST
                    blast_line = blast_content.split('\n')[0]
                    fields = blast_line.split('\t')
                    if len(fields) >= 8:
                        sstart = fields[6]  # subject start (1-based)
                        blast_start_position = int(sstart) - 1  # Convertir a 0-based
        except Exception as e:
            print(f"Warning: No se pudo leer BLAST results: {e}")
        
        # Calcular qué fragmento de referencia corresponde a la posición de match
        # USAR LA POSICIÓN DE BLAST (sstart) en lugar de first_match_position
        
        # Los fragmentos de referencia están divididos en 1000 partes iguales
        total_ref_fragments = 1000  # Ahora tenemos 1000 fragmentos de referencia
        reference_fragment_size = total_length // total_ref_fragments
        
        # Usar la posición real del BLAST para calcular el fragmento correcto
        reference_chunk_number = (blast_start_position // reference_fragment_size) + 1
        reference_chunk_number = max(1, min(total_ref_fragments, reference_chunk_number))  # Asegurar que esté en rango 1-1000
        reference_filename = f"default_part_{reference_chunk_number:04d}.fasta"
        
        # Calcular qué chunk del paciente contiene el inicio del alineamiento
        patient_chunk_size = total_length // 1000  # Tamaño de cada chunk del paciente
        patient_chunk_number = (blast_start_position // patient_chunk_size) + 1
        patient_chunk_number = max(1, min(1000, patient_chunk_number))
        
        # Usar la posición de BLAST para first_match_position también
        first_match_position = blast_start_position
        
        return {
            "chunks_available": True,
            "chunk_count": len(items),
            "first_match_position": first_match_position,
            "chunk_length": chunk_length,
            "total_length": total_length,
            "chunks_folder": prefix,
            "chunk_type": "aligned_chunks",
            "blast_results_key": blast_results_key,
            "reference_chunk_number": reference_chunk_number,
            "reference_filename": reference_filename,
            "navigation_info": {
                "match_start_position": blast_start_position,  # 0-based for frontend
                "match_end_position": blast_start_position,     # Placeholder - podríamos usar send también
                "recommended_chunk": patient_chunk_number,      # Chunk del PACIENTE que contiene el match
                "recommended_reference_chunk": reference_chunk_number,  # Chunk de REFERENCIA 
                "position_in_chunk": blast_start_position % patient_chunk_size
            }
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error leyendo chunks alineados: {e}")
            

@app.get("/pacientes/{dni}/files/{filename}", tags=["Pacientes"])
def download_fasta_file(
    dni: str,
    filename: str,
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    # Intentar primero en las carpetas de chunks según el patrón del archivo
    if filename.startswith("patient_part_") and filename.endswith(".fasta"):
        key = f"{dni}/patient_chunks/{filename}"
    elif filename.startswith("aligned_part_") and filename.endswith(".fasta"):
        key = f"{dni}/aligned_chunks/{filename}"
    else:
        key = f"{dni}/{filename}"
    
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        stream = obj["Body"]
        content_length = obj.get("ContentLength", 0)
        
        # Headers optimizados para archivos grandes
        headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(content_length),
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
        
        # Usar application/octet-stream para archivos grandes (>1MB)
        if content_length > 1024 * 1024:  # > 1MB
            media_type = "application/octet-stream"
        else:
            media_type = "text/plain"
            
        return StreamingResponse(stream, media_type=media_type, headers=headers)
    except s3_client.exceptions.NoSuchKey:
        # Si no se encuentra en la carpeta específica, intentar en otras ubicaciones
        alternative_keys = []
        
        if key.startswith(f"{dni}/patient_chunks/"):
            alternative_keys = [f"{dni}/{filename}", f"{dni}/aligned_chunks/{filename}"]
        elif key.startswith(f"{dni}/aligned_chunks/"):
            alternative_keys = [f"{dni}/{filename}", f"{dni}/patient_chunks/{filename}"]
        else:
            alternative_keys = [f"{dni}/patient_chunks/{filename}", f"{dni}/aligned_chunks/{filename}"]
        
        for alt_key in alternative_keys:
            try:
                obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=alt_key)
                stream = obj["Body"]
                content_length = obj.get("ContentLength", 0)
                
                # Headers optimizados para archivos grandes
                headers = {
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Content-Length": str(content_length),
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
                
                # Usar application/octet-stream para archivos grandes (>1MB)
                if content_length > 1024 * 1024:  # > 1MB
                    media_type = "application/octet-stream"
                elif alt_key.endswith(('.fasta', '.fa', '.fna')):
                    media_type = "text/plain"
                else:
                    media_type = "application/octet-stream"
                    
                return StreamingResponse(stream, media_type=media_type, headers=headers)
            except s3_client.exceptions.NoSuchKey:
                continue
        
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
@app.get("/pacientes/{dni}/notes", response_model=list[schemas.NoteOut], tags=["Pacientes"])
def list_notes(
    dni: str,
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    paciente = crud.get_paciente_por_dni(db, dni)
    if not paciente or paciente.doctor_id != current_doc.id:
        raise HTTPException(status_code=404, detail="Paciente no encontrado o no autorizado")
    return crud.get_notes_for_patient(db, dni)

@app.post("/pacientes/{dni}/notes", response_model=schemas.NoteOut, status_code=201, tags=["Pacientes"])
def create_note(
    dni: str,
    note_in: schemas.NoteCreate,
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    paciente = crud.get_paciente_por_dni(db, dni)
    if not paciente or paciente.doctor_id != current_doc.id:
        raise HTTPException(status_code=404, detail="Paciente no encontrado o no autorizado")
    return crud.create_note_for_patient(db, dni, note_in)

# ――― ELIMINAR un archivo FASTA del S3 ――― #
@app.delete(
    "/pacientes/{dni}/files/{filename}",        # usa {filename:path} si algún día admites subcarpetas
    status_code=204,                            # 204 No Content: OK y sin body
    tags=["Pacientes"]
)
def delete_fasta_file(
    dni: str,
    filename: str,
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """
    Borra un archivo FASTA del bucket S3 para el paciente indicado.
    El doctor autenticado debe ser el dueño del paciente.
    """
    # 1. Verificar que el paciente exista y pertenezca al doctor logueado
    paciente = crud.get_paciente_por_dni(db, dni)
    if not paciente or paciente.doctor_id != current_doc.id:
        raise HTTPException(status_code=404, detail="Paciente no encontrado o no autorizado")
    
    # 2. Construir la clave S3 y eliminar
    key = f"{dni}/{filename}"
    try:
        s3_client.delete_object(Bucket=BUCKET_NAME, Key=key)
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando de S3: {e}")
    
    # 3. 204 significa "todo bien" y no se envía contenido
    return

# ――― ELIMINAR una carpeta completa del S3 ――― #
@app.delete(
    "/pacientes/{dni}/folders/{folder_name}",
    status_code=204,
    tags=["Pacientes"]
)
def delete_folder(
    dni: str,
    folder_name: str,
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """
    Borra una carpeta completa y todos sus contenidos del bucket S3 para el paciente indicado.
    El doctor autenticado debe ser el dueño del paciente.
    """
    # 1. Verificar que el paciente exista y pertenezca al doctor logueado
    paciente = crud.get_paciente_por_dni(db, dni)
    if not paciente or paciente.doctor_id != current_doc.id:
        raise HTTPException(status_code=404, detail="Paciente no encontrado o no autorizado")
    
    # 2. Construir el prefijo de la carpeta (asegurándonos que termine con /)
    folder_prefix = f"{dni}/{folder_name}"
    if not folder_prefix.endswith('/'):
        folder_prefix += '/'
    
    try:
        # 3. Listar todos los objetos en la carpeta
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=folder_prefix)
        objects = response.get('Contents', [])
        
        if not objects:
            raise HTTPException(status_code=404, detail="Carpeta no encontrada o vacía")
        
        # 4. Eliminar todos los objetos en la carpeta
        delete_keys = [{'Key': obj['Key']} for obj in objects]
        
        # S3 permite eliminar hasta 1000 objetos en una sola operación
        while delete_keys:
            batch = delete_keys[:1000]  # Procesar en lotes de 1000
            delete_keys = delete_keys[1000:]
            
            s3_client.delete_objects(
                Bucket=BUCKET_NAME,
                Delete={'Objects': batch}
            )
        
        return  # 204 No Content
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando carpeta de S3: {e}")

def natural_key(fname: str) -> tuple[int, str]:
    """
    Extrae el último número que aparezca en el nombre (por ejemplo, 'default_part_9.fasta' → 9)
    y lo devuelve como clave de orden.  Si no hay número, usa 0.
    """
    m = re.findall(r'\d+', fname)
    num = int(m[-1]) if m else 0
    return (num, fname)    

@app.get("/split_fasta_files", tags=["Reference FASTA"])
def list_reference_fastas():
    prefix = "split_fasta_files/"
    resp   = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    items  = resp.get("Contents", [])

    # nombres sin la carpeta
    files = [
        obj["Key"].replace(prefix, "")
        for obj in items
        if obj["Key"] != prefix
    ]

    # ordenar por número (y luego por nombre)
    files.sort(key=natural_key)

    return {"files": files}


# 2) Descargar un archivo concreto para enviarlo al front-end
@app.get("/split_fasta_files/{filename}", tags=["Reference FASTA"])
def get_reference_fasta(filename: str):
    """
    Descarga un FASTA de la carpeta split_fasta_files/ y lo envía como texto.
    """
    key = f"split_fasta_files/{filename}"

    try:
        obj    = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        stream = obj["Body"]                               # streaming S3
        headers = {
            "Content-Disposition": f"attachment; filename={filename}"
        }
        return StreamingResponse(
            stream,
            media_type="text/plain",       # FASTA = texto plano
            headers=headers
        )
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Archivo de referencia no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo archivo: {e}")
    
    
@app.get("/reference-files")
def list_reference_files():
    s3 = boto3.client("s3")  # boto3 leerá las credenciales de tu entorno
    resp = s3.list_objects_v2(
        Bucket="prevenia-bucket-339712906940-1750164681373",
        Prefix="split_fasta_files/"
    )
    keys = [o["Key"] for o in resp.get("Contents", [])]
    return {"files": keys}

@app.get("/reference/cr13/status", tags=["Reference"])
def get_cr13_status():
    """Verifica el estado de la secuencia de referencia cr13"""
    global _cr13_reference_cache, _cr13_sequence_cache
    
    status = {
        "fasta_cached_in_memory": _cr13_reference_cache is not None,
        "sequence_cached_in_memory": _cr13_sequence_cache is not None,
        "local_file_exists": False,
        "s3_available": False
    }
    
    # Verificar archivo local
    try:
        with open("cr13.fasta", "r") as f:
            status["local_file_exists"] = True
            if _cr13_reference_cache is None:
                status["local_file_size"] = len(f.read())
    except FileNotFoundError:
        pass
    
    # Verificar S3 (raíz del bucket)
    try:
        s3_client.head_object(Bucket=BUCKET_NAME, Key="cr13.fasta")
        status["s3_available"] = True
    except:
        pass
    
    # Información de cache
    if _cr13_reference_cache:
        status["fasta_cache_size"] = len(_cr13_reference_cache)
    if _cr13_sequence_cache:
        status["sequence_cache_size"] = len(_cr13_sequence_cache)
    
    return status

@app.post("/reference/cr13/preload", tags=["Reference"])
def preload_cr13_reference():
    """Precarga la secuencia de referencia cr13 en memoria"""
    try:
        ref_content = get_cr13_reference()
        return {
            "message": "Secuencia cr13.fasta precargada en memoria",
            "size": len(ref_content),
            "source": "local_file" if os.path.exists("cr13.fasta") else "s3"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error precargando cr13: {e}")

@app.get("/blast/database/status", tags=["Reference"])
def get_blast_database_status():
    """Verifica el estado de la base de datos BLAST"""
    status = {
        "blast_db_exists": False,
        "cr13_fasta_exists": os.path.exists("cr13.fasta"),
        "blast_db_files": []
    }
    
    # Verificar si existen los archivos de la base de datos BLAST
    blast_db_files = [
        "blast_db/chr13_db.nhr",
        "blast_db/chr13_db.nin", 
        "blast_db/chr13_db.nsq"
    ]
    
    existing_files = []
    for db_file in blast_db_files:
        if os.path.exists(db_file):
            existing_files.append(db_file)
    
    status["blast_db_files"] = existing_files
    status["blast_db_exists"] = len(existing_files) == 3
    
    return status

@app.post("/blast/database/create", tags=["Reference"])
async def create_blast_database_endpoint():
    """Crea la base de datos BLAST manualmente"""
    try:
        await create_blast_database()
        return {
            "message": "Base de datos BLAST creada exitosamente",
            "database_path": "blast_db/chr13_db"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creando base de datos BLAST: {e}")

@app.post("/pacientes/{dni}/align_with_cr13")
async def align_sequence_with_cr13(
    dni: str = Path(..., description="DNI del paciente"),
    filename: str = Form(..., description="Nombre del archivo a alinear"),
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """Alinea una secuencia específica del paciente con cr13 usando BLAST"""
    
    # Verificar que el paciente existe y pertenece al doctor
    paciente = crud.get_paciente_por_dni(db, dni)
    if not paciente or paciente.doctor_id != current_doc.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paciente no encontrado o no te pertenece"
        )
    
    try:
        # Obtener el archivo del paciente desde S3
        patient_file_key = f"{dni}/{filename}"
        try:
            patient_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=patient_file_key)
            patient_content = patient_obj["Body"].read().decode('utf-8')
        except s3_client.exceptions.NoSuchKey:
            raise HTTPException(status_code=404, detail="Archivo del paciente no encontrado")
        
        # Crear archivo temporal para la consulta (paciente)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as query_temp:
            query_temp.write(patient_content)
            query_temp_path = query_temp.name
        
        # Crear archivo temporal para los resultados de BLAST
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as results_temp:
            results_temp_path = results_temp.name
        
        try:
            # Ejecutar BLAST
            result = subprocess.run([
                'blastn',
                '-query', query_temp_path,
                '-db', 'blast_db/chr13_db',
                '-outfmt', '6 qseqid sseqid pident length qstart qend sstart send evalue qseq',
                '-out', results_temp_path,
                '-max_target_seqs', '1',  # Solo el mejor hit
                '-evalue', '1e-5'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Error en BLAST: {result.stderr}")
            
            # Leer los resultados de BLAST
            with open(results_temp_path, 'r') as f:
                blast_results = f.read().strip()
            
            if not blast_results:
                raise HTTPException(status_code=400, detail="No se encontraron alineamientos significativos")
            
            # Parsear el primer resultado (mejor hit)
            blast_line = blast_results.split('\n')[0]
            fields = blast_line.split('\t')
            
            if len(fields) < 10:
                raise HTTPException(status_code=500, detail="Formato de resultado BLAST inválido")
            
            qseqid, sseqid, pident, length, qstart, qend, sstart, send, evalue, qseq = fields
            
            # Obtener la secuencia de referencia completa (optimizada con cache)
            ref_sequence = get_cr13_sequence()
            aligned_length = len(ref_sequence)
            
            # Calcular posición de inicio (BLAST usa 1-based, Python 0-based)
            ref_start = int(sstart) - 1
            ref_end = int(send) - 1
            
            # Mantener la secuencia query con gaps (NO remover los "-")
            query_seq_with_gaps = qseq
            
            # Crear secuencia alineada manteniendo gaps y tamaño igual a cr13
            patient_aligned_str = (
                '-' * ref_start + 
                query_seq_with_gaps + 
                '-' * max(0, aligned_length - ref_end - 1)
            )
            
            # Dividir la secuencia alineada en 1000 partes iguales
            sequence_length = len(patient_aligned_str)
            chunk_size = sequence_length // 1000
            remainder = sequence_length % 1000
            
            # Crear carpeta para las partes alineadas del paciente
            aligned_folder = f"{dni}/aligned_chunks/"
            
            # Subir cada parte a S3
            for i in range(1000):
                start_pos = i * chunk_size
                # Distribuir el remainder entre las primeras partes
                if i < remainder:
                    start_pos += i
                    end_pos = start_pos + chunk_size + 1
                else:
                    start_pos += remainder
                    end_pos = start_pos + chunk_size
                
                chunk_sequence = patient_aligned_str[start_pos:end_pos]
                
                # Crear nombre del archivo para esta parte
                chunk_filename = f"aligned_part_{i+1:04d}.fasta"  # 0001, 0002, ..., 1000
                chunk_key = f"{aligned_folder}{chunk_filename}"
                
                # Subir la parte a S3 (solo la secuencia, sin header)
                s3_client.put_object(
                    Bucket=BUCKET_NAME,
                    Key=chunk_key,
                    Body=chunk_sequence.encode('utf-8'),
                    ContentType="text/plain"
                )
            
            # Subir la secuencia alineada completa
            aligned_key = f"{dni}/aligned_{filename}"
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=aligned_key,
                Body=patient_aligned_str.encode('utf-8'),
                ContentType="text/plain"
            )
            
            # Subir los resultados de BLAST
            blast_results_key = f"{dni}/blast_results_{filename}.tsv"
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=blast_results_key,
                Body=blast_results.encode('utf-8'),
                ContentType="text/plain"
            )
            
            return JSONResponse(
                status_code=200, 
                content={
                    "message": f"Archivo {filename} alineado con cr13.fasta usando BLAST y dividido en 1000 partes con éxito",
                    "original_file": filename,
                    "aligned_key": aligned_key,
                    "blast_results_key": blast_results_key,
                    "aligned_chunks_folder": aligned_folder,
                    "sequence_length": sequence_length,
                    "chunks_created": 1000,
                    "blast_stats": {
                        "percent_identity": float(pident),
                        "alignment_length": int(length),
                        "ref_start": int(sstart),  # 1-based position in reference
                        "ref_end": int(send),      # 1-based position in reference
                        "query_start": int(qstart), # 1-based position in query
                        "query_end": int(qend),     # 1-based position in query
                        "evalue": float(evalue)
                    },
                    "alignment_method": "blast",
                    "reference_used": "blast_db/chr13_db",
                    "navigation_info": {
                        "match_start_position": ref_start,  # 0-based for frontend
                        "match_end_position": ref_end,      # 0-based for frontend
                        "recommended_chunk": ((ref_start // (sequence_length // 1000)) + 1),
                        "position_in_chunk": ref_start % (sequence_length // 1000)
                    }
                }
            )
            
        finally:
            # Limpiar archivos temporales
            import os
            try:
                os.unlink(query_temp_path)
                os.unlink(results_temp_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en el alineamiento: {e}")

@app.post("/pacientes/{dni}/blast_simple")
async def blast_simple_alignment(
    dni: str = Path(..., description="DNI del paciente"),
    filename: str = Form(..., description="Nombre del archivo a consultar"),
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """Ejecuta un BLAST simple y devuelve los resultados tabulares"""
    
    # Verificar que el paciente existe y pertenece al doctor
    paciente = crud.get_paciente_por_dni(db, dni)
    if not paciente or paciente.doctor_id != current_doc.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paciente no encontrado o no te pertenece"
        )
    
    try:
        # Obtener el archivo del paciente desde S3
        patient_file_key = f"{dni}/{filename}"
        try:
            patient_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=patient_file_key)
            patient_content = patient_obj["Body"].read().decode('utf-8')
        except s3_client.exceptions.NoSuchKey:
            raise HTTPException(status_code=404, detail="Archivo del paciente no encontrado")
        
        # Crear archivo temporal para la consulta
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as query_temp:
            query_temp.write(patient_content)
            query_temp_path = query_temp.name
        
        # Crear archivo temporal para los resultados
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as results_temp:
            results_temp_path = results_temp.name
        
        try:
            # Ejecutar BLAST
            result = subprocess.run([
                'blastn',
                '-query', query_temp_path,
                '-db', 'blast_db/chr13_db',
                '-outfmt', '6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore',
                '-out', results_temp_path,
                '-max_target_seqs', '10',  # Top 10 hits
                '-evalue', '1e-5'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Error en BLAST: {result.stderr}")
            
            # Leer los resultados
            with open(results_temp_path, 'r') as f:
                blast_results = f.read().strip()
            
            if not blast_results:
                return {
                    "message": "No se encontraron alineamientos significativos",
                    "filename": filename,
                    "results": []
                }
            
            # Parsear los resultados
            results = []
            for line in blast_results.split('\n'):
                if line.strip():
                    fields = line.split('\t')
                    if len(fields) >= 12:
                        results.append({
                            "query_id": fields[0],
                            "subject_id": fields[1],
                            "percent_identity": float(fields[2]),
                            "alignment_length": int(fields[3]),
                            "mismatches": int(fields[4]),
                            "gap_opens": int(fields[5]),
                            "query_start": int(fields[6]),
                            "query_end": int(fields[7]),
                            "subject_start": int(fields[8]),
                            "subject_end": int(fields[9]),
                            "evalue": float(fields[10]),
                            "bit_score": float(fields[11])
                        })
            
            # Subir los resultados a S3
            blast_results_key = f"{dni}/blast_simple_{filename}.tsv"
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=blast_results_key,
                Body=blast_results.encode('utf-8'),
                ContentType="text/plain"
            )
            
            return {
                "message": f"BLAST ejecutado exitosamente para {filename}",
                "filename": filename,
                "results_count": len(results),
                "results": results,
                "blast_results_key": blast_results_key,
                "alignment_method": "blast_simple"
            }
            
        finally:
            # Limpiar archivos temporales
            import os
            try:
                os.unlink(query_temp_path)
                os.unlink(results_temp_path)
            except:
                pass
                
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en BLAST simple: {e}")

@app.post("/pacientes/{dni}/process_embedding")
async def process_sequence_embedding(
    dni: str = Path(..., description="DNI del paciente"),
    filename: str = Form(..., description="Nombre del archivo FASTA a procesar"),
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """Procesa una secuencia FASTA del paciente usando la API de NVIDIA NIM para generar embeddings"""
    
    print(f"🔬 [EMBEDDING] Iniciando procesamiento de embeddings")
    print(f"   📋 DNI: {dni}")
    print(f"   📄 Archivo: {filename}")
    print(f"   👨‍⚕️ Doctor: {current_doc.nombre} (ID: {current_doc.id})")
    
    # Verificar que el paciente existe y pertenece al doctor
    print(f"🔍 [EMBEDDING] Verificando paciente...")
    paciente = crud.get_paciente_por_dni(db, dni)
    if not paciente or paciente.doctor_id != current_doc.id:
        print(f"❌ [EMBEDDING] Paciente no encontrado o no autorizado")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Paciente no encontrado o no te pertenece"
        )
    print(f"✅ [EMBEDDING] Paciente verificado: {paciente.nombres} {paciente.apellidos}")
    
    # Verificar que la clave de API existe
    print(f"🔑 [EMBEDDING] Verificando clave API...")
    if not NIM_KEY:
        print(f"❌ [EMBEDDING] Clave de API no configurada")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clave de API de NVIDIA NIM no configurada"
        )
    print(f"✅ [EMBEDDING] Clave API configurada (longitud: {len(NIM_KEY)} caracteres)")
    
    try:
        # Obtener el archivo del paciente desde S3
        patient_file_key = f"{dni}/{filename}"
        print(f"📂 [EMBEDDING] Obteniendo archivo de S3: {patient_file_key}")
        
        try:
            patient_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=patient_file_key)
            patient_content = patient_obj["Body"].read().decode('utf-8')
            content_size = len(patient_content)
            print(f"✅ [EMBEDDING] Archivo obtenido exitosamente ({content_size:,} caracteres)")
        except s3_client.exceptions.NoSuchKey:
            print(f"❌ [EMBEDDING] Archivo no encontrado en S3: {patient_file_key}")
            raise HTTPException(status_code=404, detail="Archivo del paciente no encontrado")
        except Exception as e:
            print(f"❌ [EMBEDDING] Error obteniendo archivo de S3: {e}")
            raise HTTPException(status_code=500, detail=f"Error accediendo al archivo: {e}")
        
        # Obtener la secuencia de referencia cr13
        print(f"🧬 [EMBEDDING] Obteniendo secuencia de referencia cr13...")
        chr13_seq = get_cr13_sequence()
        if not chr13_seq:
            print(f"❌ [EMBEDDING] No se pudo cargar la secuencia de referencia cr13")
            raise HTTPException(status_code=500, detail="No se pudo cargar la secuencia de referencia cr13")
        print(f"✅ [EMBEDDING] Secuencia cr13 cargada: {len(chr13_seq):,} bases")

        # Detectar si el archivo contiene mismatches con posiciones
        print(f"🧬 [EMBEDDING] Analizando contenido FASTA...")
        header_count = patient_content.count('>')
        print(f"   🏷️ Headers encontrados: {header_count}")
        
        if filename.startswith("mismatches_"):
            print(f"🎯 [EMBEDDING] Procesando archivo de mismatches con posiciones")
            mismatches = extract_mismatches_from_fasta(patient_content)
            print(f"🧬 [EMBEDDING] Mismatches extraídos: {len(mismatches)}")

            if not mismatches:
                print(f"❌ [EMBEDDING] No se pudieron extraer mismatches válidos")
                raise HTTPException(status_code=400, detail="No se pudieron extraer mismatches válidos del archivo")
                
            for i, (pos, seq) in enumerate(mismatches[:3]):  # Solo mostrar los primeros 3
                print(f"   Mismatch {i+1}: posición {pos:,}, secuencia {len(seq):,} bases")
            if len(mismatches) > 3:
                print(f"   ... y {len(mismatches) - 3} mismatches más")
        else:
            print(f"❌ [EMBEDDING] Este endpoint solo procesa archivos de mismatches (mismatches_*.fasta)")
            raise HTTPException(status_code=400, detail="Este endpoint solo procesa archivos de mismatches")
        
        # Procesar cada mismatch con la API de NVIDIA NIM
        print(f"🤖 [EMBEDDING] Iniciando procesamiento con NVIDIA NIM...")
        mutation_embeddings: list[np.ndarray] = []  # Vector de 32768 por mutación

        API_URL = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/forward"
        print(f"🔗 [EMBEDDING] URL de API: {API_URL}")
        
        try:
            for i, (mid_pos, alt_seq) in enumerate(mismatches, 1):
                print(f"🎯 [EMBEDDING] Procesando mismatch {i}/{len(mismatches)} en posición {mid_pos:,}...")
                
                # Generar las 4 secuencias para esta mutación
                print(f"   🧬 Generando 4 secuencias variantes...")
                ref_seq, var_seq, ref_seq_rev_comp, var_seq_rev_comp = parse_sequences3(mid_pos, alt_seq, chr13_seq)
                
                four_sequences = [
                    ("ref_seq", ref_seq),
                    ("var_seq", var_seq), 
                    ("ref_seq_rev_comp", ref_seq_rev_comp),
                    ("var_seq_rev_comp", var_seq_rev_comp)
                ]
                
                print(f"   📏 Longitudes de secuencias: {[len(seq[1]) for seq in four_sequences]}")
                
                # Obtener embeddings para las 4 secuencias
                sequence_embeddings = []
                
                for j, (seq_type, sequence) in enumerate(four_sequences, 1):
                    print(f"   ⚡ Procesando {seq_type} ({j}/4) - {len(sequence):,} bases...")
                    
                    start_time = time.time()
                response = requests.post(
                    url=API_URL,
                    headers={"Authorization": f"Bearer {NIM_KEY}"},
                    json={
                            "sequence": sequence,
                        "output_layers": ["blocks.24.inner_mha_cls"]
                    },
                    timeout=300  # 5 min por secuencia
                )
                    
                elapsed_time = time.time() - start_time
                print(f"      ⏱️ Tiempo: {elapsed_time:.2f}s, Status: {response.status_code}")
                    
                response.raise_for_status()

                content_type = response.headers.get('content-type', '')
                    
                if 'application/zip' in content_type:
                    zip_data = io.BytesIO(response.content)
                    with zipfile.ZipFile(zip_data, 'r') as zip_file:
                        response_file = zip_file.namelist()[0]
                        with zip_file.open(response_file) as file:
                            rj = json.loads(file.read().decode('utf-8'))
                elif 'application/json' in content_type:
                    rj = response.json()
                else:
                    raise HTTPException(500, f"Content-Type no soportado: {content_type}")

                if 'data' not in rj:
                        raise HTTPException(500, f"Campo 'data' no encontrado para {seq_type}")

                tensor_data = base64.b64decode(rj['data'])
                tensor_dict = np.load(io.BytesIO(tensor_data))
                    
                expected_key = 'blocks.24.inner_mha_cls.output'
                if expected_key not in tensor_dict:
                    raise HTTPException(500, f"Clave de embedding no encontrada para {seq_type}")

                tensor = tensor_dict[expected_key]
                print(f"      ✅ Embedding {seq_type}: shape original {tensor.shape}")
                
                # Promedio por dimensión 1 (longitud de secuencia) y luego flatten
                # De (1, seq_len, 8192) -> (1, 8192) -> (8192,)
                avg_tensor = np.mean(tensor, axis=1)  # Promedio por tokens
                print(f"      📊 Shape después de promedio axis=1: {avg_tensor.shape}")
                
                flat_tensor = avg_tensor.flatten()   # Flatten para obtener (8192,)
                print(f"      📏 Shape después de flatten: {flat_tensor.shape}")
                
                # Verificación final del tamaño
                if len(flat_tensor) != 8192:
                    raise HTTPException(500, f"Vector final inesperado para {seq_type}: {len(flat_tensor)}, esperado 8192")
                
                sequence_embeddings.append(flat_tensor)  # Vector de 8192 elementos
                print(f"      ✅ Vector {seq_type} agregado: {len(flat_tensor)} elementos")
                
                # Promediar los 4 embeddings y concatenar para formar vector de 32768
                print(f"   🧮 Promediando 4 embeddings...")
                print(f"      📊 Shapes de los 4 embeddings: {[emb.shape for emb in sequence_embeddings]}")
                
                sequence_embeddings_array = np.array(sequence_embeddings)
                print(f"      📊 Array de embeddings: shape {sequence_embeddings_array.shape}")
                
                avg_embedding = np.mean(sequence_embeddings_array, axis=0)  # Promedio de shape (8192,)
                print(f"      📊 Embedding promediado: shape {avg_embedding.shape}")
                
                # Concatenar 4 veces para formar vector de 32768
                concatenated_embedding = np.concatenate([avg_embedding] * 4)
                print(f"   🔗 Vector concatenado: shape {concatenated_embedding.shape}")
                
                mutation_embeddings.append(concatenated_embedding)

        except requests.exceptions.RequestException as e:
            print(f"❌ [EMBEDDING] Error de conexión con NVIDIA NIM: {e}")
            raise HTTPException(500, f"Error al procesar con la API de NVIDIA NIM: {str(e)}")
        except Exception as e:
            print(f"❌ [EMBEDDING] Error procesando respuesta de API: {e}")
            raise HTTPException(500, f"Error procesando la respuesta de la API: {str(e)}")

        print(f"🧮 [EMBEDDING] Embeddings de mutaciones obtenidos: {len(mutation_embeddings)}")
        if not mutation_embeddings:
            print(f"❌ [EMBEDDING] No se obtuvieron embeddings")
            raise HTTPException(500, "No se pudo obtener embeddings de la API")

        # Verificar que todos los embeddings tienen el tamaño correcto (32768)
        print(f"🔢 [EMBEDDING] Verificando embeddings de mutaciones...")
        for i, embedding in enumerate(mutation_embeddings):
            print(f"   Mutación {i+1}: shape {embedding.shape}")
            if embedding.shape[0] != 32768:
                print(f"⚠️ [EMBEDDING] Tamaño inesperado en mutación {i+1}: {embedding.shape[0]} != 32768")

        # Promediar todos los embeddings de mutaciones para obtener el resultado final
        print(f"🧮 [EMBEDDING] Promediando {len(mutation_embeddings)} embeddings de mutaciones...")
        final_embedding = np.mean(mutation_embeddings, axis=0)
        print(f"   Embedding final: shape {final_embedding.shape}, dtype {final_embedding.dtype}")

        # Guardar el embedding en S3 como archivo pickle
        print(f"💾 [EMBEDDING] Guardando embedding en S3...")
        try:
            # Serializar el embedding final
            print(f"   🔄 Serializando embedding final con pickle...")
            embedding_data = pickle.dumps(final_embedding)
            embedding_size = len(embedding_data)
            print(f"   📦 Datos serializados: {embedding_size:,} bytes ({embedding_size/(1024*1024):.2f} MB)")
            
            # Crear nombre del archivo de embedding
            base_filename = filename.rsplit('.', 1)[0]  # Quitar extensión
            embedding_filename = f"embedding_paciente_{base_filename}.pkl"
            embedding_key = f"{dni}/{embedding_filename}"  # Guardar en la raíz de la carpeta del paciente
            print(f"   🗂️ Archivo destino: {embedding_key}")
            
            # Subir a S3
            print(f"   ☁️ Subiendo a S3...")
            s3_client.put_object(
                Bucket=BUCKET_NAME,
                Key=embedding_key,
                Body=embedding_data,
                ContentType="application/octet-stream"
            )
            print(f"   ✅ Embedding guardado exitosamente en S3")
            
            result = {
                    "message": f"Embedding generado exitosamente para {filename}",
                    "original_file": filename,
                    "embedding_file": embedding_filename,
                    "embedding_key": embedding_key,
                "mutations_processed": len(mismatches),
                "sequences_per_mutation": 4,
                "total_api_calls": len(mismatches) * 4,
                "embedding_shape": list(final_embedding.shape),
                "embedding_size_mb": round(embedding_size/(1024*1024), 2),
                "api_used": "NVIDIA NIM EVO-2 40B",
                "processing_method": "4_sequences_per_mutation_averaged_and_concatenated"
            }
            
            print(f"🎉 [EMBEDDING] Procesamiento completado exitosamente!")
            print(f"   📄 Archivo original: {filename}")
            print(f"   🎯 Mutaciones procesadas: {len(mismatches)}")
            print(f"   🧬 Total llamadas a API: {len(mismatches) * 4}")
            print(f"   📊 Forma del embedding final: {final_embedding.shape}")
            print(f"   💾 Tamaño: {result['embedding_size_mb']} MB")
            
            return JSONResponse(status_code=200, content=result)
            
        except Exception as e:
            print(f"❌ [EMBEDDING] Error guardando en S3: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error guardando embedding en S3: {str(e)}"
            )
            
    except HTTPException as e:
        print(f"❌ [EMBEDDING] HTTPException: {e.detail}")
        raise
    except Exception as e:
        print(f"💥 [EMBEDDING] Error inesperado: {e}")
        import traceback
        print(f"📋 [EMBEDDING] Traceback completo:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error inesperado: {str(e)}")

@app.get("/pacientes/stats/monthly_new", tags=["Estadísticas"])
def monthly_new_patients(
    year: int = Query(..., description="Año a consultar, e.g. 2025"),
    db: Session = Depends(get_db)
):
    """
    Devuelve una lista de 12 enteros: número de pacientes creados en cada mes
    del año indicado (enero = índice 0, diciembre = índice 11).
    """
    # Consulta agrupada por mes
    rows = (
        db.query(
            extract("month", models.Paciente.created_at).label("mes"),
            func.count(models.Paciente.id).label("cantidad")
        )
        .filter(extract("year", models.Paciente.created_at) == year)
        .group_by("mes")
        .order_by("mes")
        .all()
    )

    stats = [0] * 12
    for mes, cantidad in rows:
        stats[int(mes) - 1] = cantidad

    return {"year": year, "monthly_new": stats}

@app.get("/pacientes/stats/years", tags=["Estadísticas"])
def available_years(db: Session = Depends(get_db)):
    """
    Devuelve la lista de años en los que hay pacientes creados,
    ordenados ascendentemente.
    """
    rows = (
        db.query(distinct(extract("year", models.Paciente.created_at)))
          .order_by(extract("year", models.Paciente.created_at))
          .all()
    )
    years = [int(y[0]) for y in rows if y[0] is not None]
    return {"years": years}


@app.get("/pacientes/stats/active_inactive", tags=["Estadísticas"])
def active_inactive(
    days: int = Query(30, description="Número de días para considerar actividad"),
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):

    cutoff = datetime.utcnow() - timedelta(days=days)

    recent_dnis = (
        db.query(models.Note.patient_dni)
          .filter(models.Note.timestamp >= cutoff)
          .distinct()
          .subquery()
    )

    total = db.query(models.Paciente).filter(models.Paciente.doctor_id == current_doc.id).count()
    activos = (
        db.query(models.Paciente)
          .filter(
            models.Paciente.doctor_id == current_doc.id,
            models.Paciente.dni.in_(recent_dnis)
          )
          .count()
    )
    inactivos = total - activos

    return {"total": total, "active": activos, "inactive": inactivos}


def get_bucket_size_gb(bucket_name: str, region_name: str = "us-east-1") -> float | None:
    cloudwatch = boto3.client("cloudwatch", region_name=region_name)
    resp = cloudwatch.get_metric_statistics(
        Namespace="AWS/S3",
        MetricName="BucketSizeBytes",
        Dimensions=[
            {"Name": "BucketName", "Value": bucket_name},
            {"Name": "StorageType", "Value": "StandardStorage"},
        ],
        StartTime=datetime.utcnow() - timedelta(days=2),
        EndTime=datetime.utcnow(),
        Period=86400,  # segundos en un día
        Statistics=["Average"],
    )
    dps = resp.get("Datapoints", [])
    if not dps:
        return None
    # Tomamos el datapoint más reciente
    latest = max(dps, key=lambda x: x["Timestamp"])
    size_bytes = latest["Average"]
    return round(size_bytes / (1024**3), 2)


@app.get("/stats/bucket_usage", tags=["Estadísticas"])
def bucket_usage():
    bucket = "prevenia-bucket"  # o lee de tus env vars
    used_gb = get_bucket_size_gb(bucket)
    if used_gb is None:
        raise HTTPException(
            status_code=404, detail="No hay datos de uso en CloudWatch para este bucket"
        )
    return {"used_gb": used_gb}



@app.get("/visualizacion/files", tags=["Visualizacion3D"])
def list_visualizacion_files():

    prefix = "visualizacion/"
    try:
        resp = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        items = resp.get("Contents", [])
        # Filtramos la propia carpeta y devolvemos sólo el nombre del fichero
        files = [
            obj["Key"].replace(prefix, "", 1)
            for obj in items
            if obj["Key"] != prefix and obj["Key"].endswith(".cif")
        ]
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listando archivos: {e}")

@app.get("/visualizacion/{filename}", tags=["Visualizacion3D"])
def get_visualizacion_file(filename: str):
    key = f"visualizacion/{filename}"
    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
        body = obj["Body"].read()
        return StreamingResponse(
            io.BytesIO(body),
            media_type="chemical/x-cif",
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )
    except s3_client.exceptions.NoSuchKey:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error descargando archivo: {e}")
    
### ——— RUTAS PARA CALENDARIO ——— ###

@app.get("/calendario/dia/{fecha}", tags=["Calendario"])
def list_appointments_by_day(
    fecha: str,  # "YYYY-MM-DD"
    current_doc: models.Doctor = Depends(auth.get_current_doctor),
    db: Session = Depends(get_db)
):
    """
    Devuelve todas las citas (eventos o visitas) de este doctor en la fecha indicada.
    """
    try:
        dt = datetime.fromisoformat(fecha)
    except ValueError:
        raise HTTPException(400, "Formato de fecha inválido, use YYYY-MM-DD")
    start = datetime(dt.year, dt.month, dt.day)
    end   = start + timedelta(days=1)
    citas = (
        db.query(models.Appointment)
          .filter(
            models.Appointment.doctor_id == current_doc.id,
            models.Appointment.fecha_hora >= start,
            models.Appointment.fecha_hora < end
          )
          .all()
    )
    return citas


@app.post("/calendario/evento", response_model=schemas.AppointmentOut, status_code=201, tags=["Calendario"])
def create_evento(
    calendario: schemas.AppointmentCreate,  # { fecha_hora, asunto, lugar, descripcion }
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """
    Crea una nueva nota/recordatorio en el calendario del doctor autenticado.
    """
    return crud.create_appointment_for_doctor(db, current_doc.id, calendario)



@app.delete("/calendario/{appointment_id}", status_code=204, tags=["Calendario"])
def delete_appointment(
    appointment_id: int,
    db: Session = Depends(get_db),
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """
    Borra la cita del doctor autenticado.
    """
    appt = crud.get_appointment(db, appointment_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Cita no encontrada")
    if appt.doctor_id != current_doc.id:
        raise HTTPException(status_code=403, detail="No autorizado")
    crud.delete_appointment(db, appointment_id)
    return Response(status_code=204)

@app.head("/pacientes/{dni}/files/{filename}", tags=["Pacientes"])
def get_file_info(
    dni: str,
    filename: str,
    current_doc: models.Doctor = Depends(auth.get_current_doctor)
):
    """
    Obtiene información del archivo (tamaño, tipo, etc.) sin descargarlo.
    Útil para verificar archivos grandes antes de la descarga.
    """
    # Determinar la ubicación del archivo según su patrón
    if filename.startswith("patient_part_") and filename.endswith(".fasta"):
        key = f"{dni}/patient_chunks/{filename}"
    elif filename.startswith("aligned_part_") and filename.endswith(".fasta"):
        key = f"{dni}/aligned_chunks/{filename}"
    else:
        key = f"{dni}/{filename}"
    
    try:
        # Usar head_object para obtener metadatos sin descargar el archivo
        response = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
        content_length = response.get("ContentLength", 0)
        last_modified = response.get("LastModified")
        
        # Headers con información del archivo
        headers = {
            "Content-Length": str(content_length),
            "Last-Modified": last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT") if last_modified else "",
            "X-File-Size-MB": str(round(content_length / (1024 * 1024), 2)),
            "X-File-Type": "large" if content_length > 1024 * 1024 else "small"
        }
        
        return Response(status_code=200, headers=headers)
        
    except s3_client.exceptions.NoSuchKey:
        # Intentar ubicaciones alternativas
        alternative_keys = []
        
        if key.startswith(f"{dni}/patient_chunks/"):
            alternative_keys = [f"{dni}/{filename}", f"{dni}/aligned_chunks/{filename}"]
        elif key.startswith(f"{dni}/aligned_chunks/"):
            alternative_keys = [f"{dni}/{filename}", f"{dni}/patient_chunks/{filename}"]
        else:
            alternative_keys = [f"{dni}/patient_chunks/{filename}", f"{dni}/aligned_chunks/{filename}"]
        
        for alt_key in alternative_keys:
            try:
                response = s3_client.head_object(Bucket=BUCKET_NAME, Key=alt_key)
                content_length = response.get("ContentLength", 0)
                last_modified = response.get("LastModified")
                
                headers = {
                    "Content-Length": str(content_length),
                    "Last-Modified": last_modified.strftime("%a, %d %b %Y %H:%M:%S GMT") if last_modified else "",
                    "X-File-Size-MB": str(round(content_length / (1024 * 1024), 2)),
                    "X-File-Type": "large" if content_length > 1024 * 1024 else "small"
                }
                
                return Response(status_code=200, headers=headers)
                
            except s3_client.exceptions.NoSuchKey:
                continue
        
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo información del archivo: {e}")
