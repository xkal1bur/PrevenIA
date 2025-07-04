FROM python:3.10

RUN apt-get update && apt-get install -y unzip && apt-get clean

COPY ncbi-blast-2.16.0+/bin /blast/bin



ENV PATH="/blast/bin:$PATH"

WORKDIR /app/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
