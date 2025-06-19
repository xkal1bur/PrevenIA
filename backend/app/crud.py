from sqlalchemy.orm import Session
import models as models, schemas, auth as auth
import os
import shutil
from fastapi import UploadFile
from typing import Optional, List
from datetime import date, datetime

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

def get_paciente_por_correo(db: Session, correo: str):
    return db.query(models.Paciente).filter(models.Paciente.correo == correo).first()

def get_paciente_por_id(db: Session, paciente_id: int):
    return db.query(models.Paciente).filter(models.Paciente.id == paciente_id).first()

def get_paciente_por_dni(db: Session, dni: str) -> Optional[models.Paciente]:
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

def get_pacientes_por_doctor(db: Session, doctor_id: int) -> List[models.Paciente]:
    return (
        db.query(models.Paciente)
          .filter(models.Paciente.doctor_id == doctor_id)
          .all()
    )
    
def get_notes_for_patient(db: Session, dni: str):
    return db.query(models.Note).filter(models.Note.patient_dni == dni).order_by(models.Note.timestamp.desc()).all()

def create_note_for_patient(db: Session, dni: str, note: schemas.NoteCreate):
    db_note = models.Note(
        patient_dni=dni,
        title=note.title,
        content=note.content
    )
    db.add(db_note)
    db.commit()
    db.refresh(db_note)
    return db_note



def get_appointments_for_doctor_and_day(
    db: Session,
    doctor_id: int,
    day: date
) -> list[models.Appointment]:
    """
    Devuelve todas las citas de un doctor en la fecha indicada.
    """
    # Convertimos la fecha a rango [00:00, 23:59:59]
    start = datetime.combine(day, datetime.min.time())
    end   = datetime.combine(day, datetime.max.time())
    return (
        db.query(models.Appointment)
          .filter(models.Appointment.doctor_id == doctor_id)
          .filter(models.Appointment.fecha_hora >= start)
          .filter(models.Appointment.fecha_hora <= end)
          .order_by(models.Appointment.fecha_hora)
          .all()
    )

def create_appointment_for_doctor(
    db: Session,
    doctor_id: int,
    appointment_in: schemas.AppointmentCreate
) -> models.Appointment:
    """
    Crea una nueva cita/nota para un doctor.
    """
    db_obj = models.Appointment(
        doctor_id   = doctor_id,
        fecha_hora  = appointment_in.fecha_hora,
        asunto      = appointment_in.asunto,
        lugar       = appointment_in.lugar,
        descripcion = appointment_in.descripcion
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj

def get_appointment(db: Session, appointment_id: int) -> Optional[models.Appointment]:
    return db.query(models.Appointment).get(appointment_id)

def delete_appointment(db: Session, appointment_id: int):
    db.query(models.Appointment).filter(models.Appointment.id == appointment_id).delete()
    db.commit()
