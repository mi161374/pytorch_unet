FROM python:3.11.4

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# For Ubuntu-based image
RUN apt-get update && apt-get install -y libgl1-mesa-glx


EXPOSE 5000

CMD ["python", "app.py"]
