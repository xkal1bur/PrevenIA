import os
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
import boto3
import models, schemas, crud, auth
from database import SessionLocal, engine
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="API del Prevenia")

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

BUCKET_NAME = "prevenia-bucket-767397661639-1748963781163"


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


@app.post("/pacientes/{dni}/upload_fasta")
async def upload_fasta(
    dni: str = Path(..., description="DNI del paciente"),
    fasta_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    
    if not fasta_file.filename.lower().endswith((".fasta", ".fa", ".fna")):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .fasta/.fa/.fna")

    key = f"{dni}/{fasta_file.filename}"

    try:
        contents = await fasta_file.read()

        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=key,
            Body=contents,
            ContentType="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error subiendo a S3: {e}")

    return JSONResponse(status_code=200, content={"message": "Archivo subido con éxito", "s3_key": key})