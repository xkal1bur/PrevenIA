from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any, List

class DoctorCreate(BaseModel):
    nombre: str
    especialidad: str
    correo: EmailStr
    password: str
    clinic_name: str

# Esquema para serializar la respuesta después de crear un doctor:
class DoctorOut(BaseModel):
    id: int
    nombre: str
    especialidad: str
    correo: EmailStr
    creado_el: datetime
    clinic_name: str

    class Config:
        from_attributes = True  

# Esquema para el token:
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    correo: Optional[EmailStr] = None

class DoctorLogin(BaseModel):
    correo: EmailStr
    password: str

class DoctorProfile(BaseModel):
    id: int
    nombre: str
    clinic_name: str

    class Config:
        from_attributes = True

class PacienteCreate(BaseModel):
    dni: str
    nombres: str
    apellidos: str
    edad: int
    celular: str
    correo: EmailStr
    doctor_id: int     

class PacienteOut(BaseModel):
    id: int
    dni: str
    nombres: str
    apellidos: str
    edad: int
    celular: str
    correo: EmailStr
    foto: Optional[str]
    doctor_id: int

    class Config:
        from_attributes = True


# ——— ESQUEMA PARA PREDICCIONES DE ML ——— #

class PatientInfo(BaseModel):
    dni: str
    name: str

class PredictionsResponse(BaseModel):
    status: str
    total_models: int
    patient_info: PatientInfo
    sample_used: str
    predictions: Dict[str, Dict[str, Any]]
    description: str