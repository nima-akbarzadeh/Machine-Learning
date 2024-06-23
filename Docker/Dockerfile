FROM python:3.9

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Make port 80 available to the world outside this container
EXPOSE 80

CMD exec gunicorn --bind=:80 --workers=1 --threads=1 --timeout=1800 --log-file=- --access-logfile=- --log-level=debug main:app
