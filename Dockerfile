
FROM python:3.14-slim

WORKDIR /app

# Install the application and its dependencies.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# The trained model will be mounted by Docker Compose.
ENV SAT_GUARD_MODEL_PATH=/app/artifacts/cadc0872_random_forest.joblib

EXPOSE 8000

CMD ["uvicorn", "sat_guard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]