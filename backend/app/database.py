import os
import json
import boto3
from botocore.exceptions import ClientError

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION")
SECRET_ARN = os.getenv("AURORA_SECRET_ARN")

if not AWS_REGION or not SECRET_ARN:
    raise RuntimeError("Faltan AWS_REGION o AURORA_SECRET_ARN en variables de entorno.")

def get_secret_dict():
    session = boto3.session.Session(region_name=AWS_REGION)
    client = session.client("secretsmanager")
    try:
        resp = client.get_secret_value(SecretId=SECRET_ARN)
    except ClientError as e:
        raise RuntimeError(f"Error al obtener el secret de Secrets Manager: {e}")

    if "SecretString" in resp:
        secret_str = resp["SecretString"]
        try:
            secret_dict = json.loads(secret_str)
            return secret_dict
        except json.JSONDecodeError:
            raise RuntimeError("El formato del SecretString no es JSON válido.")
    else:
        secret_bytes = resp["SecretBinary"]
        secret_dict = json.loads(secret_bytes.decode("utf-8"))
        return secret_dict

# Obtenemos el diccionario con las credenciales
_secret = get_secret_dict()

# Extraemos cada campo
username = _secret.get("username")
password = _secret.get("password")
dbname   = _secret.get("dbname")
host     = _secret.get("host") or os.getenv("AURORA_HOST")
port     = _secret.get("port") or os.getenv("AURORA_PORT", "5432")

# Aquí imprimimos cada uno en consola
print("=== Credenciales Aurora extraídas ===")
print("username:", username)
print("password:", password)
print("dbname:  ", dbname)
print("host:    ", host)
print("port:    ", port)
print("=====================================")

if not all([username, password, dbname, host, port]):
    raise RuntimeError("Faltan campos obligatorios en el secret de Aurora (username/password/dbname/host/port).")

DATABASE_URL = f"postgresql://{username}:{password}@{host}:{port}/{dbname}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
