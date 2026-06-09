# Usa una imagen oficial de Python 3.11 (más estable para librerías de IA)
FROM python:3.11-slim

# Crea el usuario no root requerido por Hugging Face
RUN useradd -m -u 1000 user



# Cambia al usuario creado
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos de requerimientos primero
COPY --chown=user ./requirements.txt requirements.txt

# INSTALA PYTORCH CPU PRIMERO: Esto evita que intente descargar la versión GPU de 3 GB y rompa la memoria del servidor de Hugging Face
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn python-multipart jinja2 ultralytics

# Ultralytics fuerza la instalación de opencv-python (la versión que necesita GUI). 
# La desinstalamos para que el sistema use opencv-python-headless que instalamos en requirements.txt
RUN pip uninstall -y opencv-python

# Copia el resto de los archivos del proyecto dándole permisos al usuario
COPY --chown=user . /app

# Hugging Face requiere obligatoriamente exponer el puerto 7860
EXPOSE 7860

# Arranca la aplicación en el puerto 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

