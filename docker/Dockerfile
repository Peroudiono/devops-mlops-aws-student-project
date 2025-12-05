# 1. Image Python officielle
FROM python:3.10-slim

# 2. Créer un dossier dans le conteneur
WORKDIR /app

# 3. Copier les fichiers du projet
COPY ../api ./api
COPY ../model ./model
COPY ../requirements.txt ./requirements.txt

# 4. Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# 5. Exposer le port Flask
EXPOSE 5000

# 6. Lancer l'application
CMD ["python", "api/app.py"]
