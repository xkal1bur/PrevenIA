from sqlalchemy.orm import Session
import models as models, schemas, auth as auth
import os
import shutil
from fastapi import UploadFile

def get_doctor_by_email(db: Session, correo: str):
    return db.query(models.Doctor).filter(models.Doctor.correo == correo).first()

def create_doctor(db: Session, doc: schemas.DoctorCreate):
    hashed = auth.hash_password(doc.password)
    db_doc = models.Doctor(
        nombre=doc.nombre,
        especialidad=doc.especialidad,
        correo=doc.correo,
        hashed_pw=hashed,
        clinic_name  = doc.clinic_name
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc

def get_paciente_por_dni(db: Session, dni: str):
    return db.query(models.Paciente).filter(models.Paciente.dni == dni).first()

def get_paciente_por_correo(db: Session, correo: str):
    return db.query(models.Paciente).filter(models.Paciente.correo == correo).first()

def create_paciente(db: Session, paciente: schemas.PacienteCreate, foto_file: UploadFile) -> models.Paciente:

    carpeta_destino = "static/pacientes"
    os.makedirs(carpeta_destino, exist_ok=True)

    filename = f"{paciente.dni}_{foto_file.filename}"
    filepath = os.path.join(carpeta_destino, filename)

    # Guardar el archivo físico
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(foto_file.file, buffer)

    db_paciente = models.Paciente(
        dni=paciente.dni,
        nombres=paciente.nombres,
        apellidos=paciente.apellidos,
        edad=paciente.edad,
        celular=paciente.celular,
        correo=paciente.correo,
        foto=filename  
    )
    db.add(db_paciente)
    db.commit()
    db.refresh(db_paciente)
    return db_paciente

def get_paciente_por_id(db: Session, paciente_id: int):
    return db.query(models.Paciente).filter(models.Paciente.id == paciente_id).first()

def get_paciente_por_dni(db: Session, dni: str):
    return db.query(models.Paciente).filter(models.Paciente.dni == dni).first()

def get_paciente_por_correo(db: Session, correo: str):
    return db.query(models.Paciente).filter(models.Paciente.correo == correo).first()

def get_pacientes(db: Session, skip: int = 0, limit: int = 100):
    return db.query(models.Paciente).offset(skip).limit(limit).all()

def create_paciente(db: Session, paciente: schemas.PacienteCreate, foto_file: UploadFile) -> models.Paciente:
    carpeta_destino = "static/pacientes"
    os.makedirs(carpeta_destino, exist_ok=True)

    filename = f"{paciente.dni}_{foto_file.filename}"
    filepath = os.path.join(carpeta_destino, filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(foto_file.file, buffer)

    db_paciente = models.Paciente(
        dni=paciente.dni,
        nombres=paciente.nombres,
        apellidos=paciente.apellidos,
        edad=paciente.edad,
        celular=paciente.celular,
        correo=paciente.correo,
        foto=filename,
        doctor_id=paciente.doctor_id
    )
    db.add(db_paciente)
    db.commit()
    db.refresh(db_paciente)
    return db_paciente

def get_pacientes_por_doctor(db: Session, doctor_id: int) -> list[models.Paciente]:
    return (
        db.query(models.Paciente)
          .filter(models.Paciente.doctor_id == doctor_id)
          .all()
    )
    
def get_paciente_por_dni(db: Session, dni: str) -> models.Paciente | None:
    return db.query(models.Paciente).filter(models.Paciente.dni == dni).first()