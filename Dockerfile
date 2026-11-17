# Project Alpha — app (dashboard + motore) in un container.
# Pensato per un piccolo server Linux con Tailscale: il container usa la rete
# dell'host e l'app ascolta solo su 127.0.0.1; `tailscale serve` sull'host la
# rende raggiungibile solo dai tuoi dispositivi.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Europe/Rome

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . && useradd --create-home alpha
COPY configs ./configs
COPY reports/lab ./reports/lab
COPY reports/research ./reports/research

# Dati e risultati vivono su volumi persistenti (vedi deploy/compose.yaml).
RUN mkdir -p data reports/paper reports/paper_permanente && chown -R alpha:alpha /app
USER alpha

CMD ["project-alpha", "app", "--port", "8050"]
