import boto3

s3 = boto3.client("s3")

bucket_name = "prevenia-bucket-971403199924-1751551489647"
archivo_s3 = "cr13.fasta"
archivo_local = "cr13.fasta"

s3.download_file(bucket_name, archivo_s3, archivo_local)

print(f"✅ Archivo '{archivo_s3}' descargado exitosamente como '{archivo_local}'")
