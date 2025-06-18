import os
import json
import boto3
from botocore.exceptions import ClientError

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from dotenv import load_dotenv

load_dotenv()

def try_aurora_connection():
    """Intenta conectar con Aurora usando AWS Secrets Manager"""
    try:
        AWS_REGION = os.getenv("AWS_REGION")
        SECRET_ARN = os.getenv("AURORA_SECRET_ARN")

        if not AWS_REGION or not SECRET_ARN:
            print("🔍 Variables de Aurora no encontradas, usando base de datos local...")
            return None

        print("🔍 Intentando conexión con Aurora...")
        
        session = boto3.session.Session(region_name=AWS_REGION)
        client = session.client("secretsmanager")
        
        resp = client.get_secret_value(SecretId=SECRET_ARN)
        
        if "SecretString" in resp:
            secret_str = resp["SecretString"]
            secret_dict = json.loads(secret_str)
        else:
            secret_bytes = resp["SecretBinary"]
            secret_dict = json.loads(secret_bytes.decode("utf-8"))

        # Extraer credenciales
        username = secret_dict.get("username")
        password = secret_dict.get("password")
        dbname = secret_dict.get("dbname")
        host = secret_dict.get("host") or os.getenv("AURORA_HOST")
        port = secret_dict.get("port") or os.getenv("AURORA_PORT", "5432")

        if not all([username, password, dbname, host, port]):
            print("⚠️ Credenciales de Aurora incompletas, usando base de datos local...")
            return None

        database_url = f"postgresql://{username}:{password}@{host}:{port}/{dbname}"
        print("✅ Conexión con Aurora configurada correctamente")
        return database_url

    except ClientError as e:
        print(f"⚠️ Error al conectar con Aurora: {e}")
        print("🔄 Fallback a base de datos local...")
        return None
    except Exception as e:
        print(f"⚠️ Error inesperado con Aurora: {e}")
        print("🔄 Fallback a base de datos local...")
        return None

def get_database_url():
    """Obtiene la URL de la base de datos, intentando Aurora primero y luego local"""
    
    # Intentar Aurora primero
    aurora_url = try_aurora_connection()
    if aurora_url:
        return aurora_url
    
    # Fallback a base de datos local
    local_url = os.getenv("DATABASE_URL_LOCAL")
    if local_url:
        print("✅ Usando base de datos local desde DATABASE_URL_LOCAL")
        return local_url
    
    # Fallback final a SQLite local si no hay DATABASE_URL_LOCAL
    print("⚠️ DATABASE_URL_LOCAL no encontrada, usando SQLite local...")
    return "sqlite:///./local_database.db"

# Obtener la URL de la base de datos
DATABASE_URL = get_database_url()

print(f"🗃️ Base de datos configurada: {DATABASE_URL.split('@')[0]}@[HOST_OCULTO]" if '@' in DATABASE_URL else f"🗃️ Base de datos configurada: {DATABASE_URL}")

# Crear engine con configuración apropiada
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Generador para obtener sesiones de base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
