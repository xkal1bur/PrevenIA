from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text
from sqlalchemy.orm import relationship
from database import Base
from datetime import datetime

class Doctor(Base):
    __tablename__ = "doctor"
    id          = Column(Integer, primary_key=True, index=True)
    nombre      = Column(String, index=True)
    especialidad  = Column(String, nullable=True)
    correo      = Column(String, unique=True, index=True)
    hashed_pw   = Column(String)
    creado_el     = Column(DateTime, default=datetime.utcnow)
    clinic_name = Column(String)

    pacientes = relationship("Paciente", back_populates="doctor")
    appointments = relationship("Appointment", back_populates="doctor", cascade="all, delete-orphan")

class Paciente(Base):
    __tablename__ = "paciente"
    id        = Column(Integer, primary_key=True, index=True)
    dni       = Column(String, unique=True, index=True)
    nombres   = Column(String, index=True)
    apellidos = Column(String, index=True)
    edad      = Column(Integer)
    celular   = Column(String)
    correo    = Column(String, unique=True, index=True)
    foto      = Column(String, nullable=True)  
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)  # ← nuevo

    doctor_id = Column(Integer, ForeignKey("doctor.id"), nullable=False)
    doctor = relationship("Doctor", back_populates="pacientes")
    notes = relationship("Note", back_populates="paciente", cascade="all, delete-orphan")

class Note(Base):
    __tablename__ = "notes"

    id = Column(Integer, primary_key=True, index=True)
    patient_dni = Column(String, ForeignKey("paciente.dni"), nullable=False, index=True)
    title       = Column(String(200), nullable=False)
    content     = Column(Text, nullable=False)
    timestamp   = Column(DateTime, default=datetime.utcnow, nullable=False)

    paciente = relationship("Paciente", back_populates="notes")
    
class Appointment(Base):
    __tablename__ = "appointment"
    id            = Column(Integer, primary_key=True, index=True)
    doctor_id     = Column(Integer, ForeignKey("doctor.id"), nullable=False)
    fecha_hora    = Column(DateTime, nullable=False, index=True)
    asunto        = Column(String(200), nullable=False)
    lugar         = Column(String(200), nullable=False)
    descripcion   = Column(Text, nullable=True)
    
    doctor        = relationship("Doctor", back_populates="appointments")
