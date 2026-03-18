# =============================================================================
# Dockerfile — Insight Commerce · Recsys API
# Base  : python:3.10-slim (imagen mínima oficial; sin Debian completo)
# Target: AWS Fargate (linux/amd64 — especificar --platform en builds ARM)
#
# Cambios en esta versión (production-ready):
#   - ENV PYTHONPATH=/app garantiza que `from src.features...` y `from src.api...`
#     se resuelvan correctamente independientemente del directorio de trabajo.
#   - requirements-api.txt en el stage builder (nombre real del archivo subido).
#   - CMD apunta explícitamente a src.api.main:app.
#   - Usuario no-root (appuser) con /tmp/ escribible para los artefactos de S3.
# =============================================================================

# ── Stage 1: builder (compilación de dependencias) ────────────────────────────
# Separar la instalación de dependencias del código fuente maximiza el cache:
# si solo cambia src/, esta capa no se reconstruye en el próximo build.
FROM python:3.10-slim AS builder

WORKDIR /build

# gcc y libpq-dev son necesarios para compilar psycopg2 desde fuente.
# Se instalan y se limpia la caché de apt en el mismo RUN para minimizar la capa.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .

# --no-cache-dir: evita almacenar la caché de pip dentro de la imagen.
# --upgrade pip : usa el resolver moderno de pip (compatible con requirements complejos).
RUN pip install --upgrade pip --no-cache-dir \
    && pip install --no-cache-dir -r requirements-api.txt


# ── Stage 2: runtime (imagen final mínima) ────────────────────────────────────
FROM python:3.10-slim AS runtime

# libpq5 es la librería de runtime de PostgreSQL requerida por psycopg2.
# Solo se instala el runtime (sin gcc ni headers de compilación).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia solo los paquetes Python instalados desde el stage builder.
# Esto excluye gcc, libpq-dev y todo el toolchain de compilación.
COPY --from=builder /usr/local/lib/python3.10/site-packages \
                    /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copia manteniendo la estructura original dentro de /app
COPY src/api/ src/api/
COPY src/features/ src/features/
COPY requirements-api.txt .

# ── Seguridad: usuario no-root ────────────────────────────────────────────────
# Fargate no necesita root. Correr como usuario sin privilegios reduce
# la superficie de ataque ante vulnerabilidades de escape de contenedor.
# /tmp/ es propiedad de root con sticky bit — cualquier usuario puede escribir ahí,
# por lo que appuser puede descargar los artefactos de S3 sin necesitar permisos
# adicionales sobre /tmp/.
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Variables de entorno del proceso ─────────────────────────────────────────
# PYTHONUNBUFFERED=1   : stdout/stderr sin buffering → CloudWatch recibe logs en tiempo real.
# PYTHONDONTWRITEBYTECODE=1 : no genera archivos .pyc en el contenedor (FS read-only).
# PYTHONPATH=/app      : CRÍTICO — permite que `from src.features.feature_engineering import ...`
#                        y `from src.api.inference import ...` se resuelvan correctamente
#                        desde cualquier directorio de trabajo, incluyendo el que usa uvicorn.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Puerto expuesto — debe coincidir con containerPort en la Task Definition
# y con el puerto del Target Group del ALB.
EXPOSE 8000

# ── Comando de arranque ───────────────────────────────────────────────────────
# src.api.main:app  → módulo Python resuelto desde PYTHONPATH=/app
#                     equivalente a /app/src/api/main.py → objeto `app`
# --host 0.0.0.0   → escucha en todas las interfaces (requerido en contenedores)
# --port 8000      → coincide con EXPOSE y con el Target Group del ALB
# --workers 1      → un worker por tarea Fargate; escalar añadiendo tareas (ECS)
# --log-level info → uvicorn loguea INFO; los logs detallados vienen del logger "api"
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]