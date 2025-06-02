from sqlalchemy import Column, Integer, String, ForeignKey, DateTime
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

    doctor_id = Column(Integer, ForeignKey("doctor.id"), nullable=False)
    doctor = relationship("Doctor", back_populates="pacientes")
