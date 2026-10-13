FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LOCALDESK_REGISTRY=/models
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir '.[service]' && useradd --uid 10001 --create-home app
# Ready to serve: the packaged IT-tickets demo as a verified, promoted registry version.
# Mount your own registry over /models to replace it.
COPY docker/demo-registry /models
USER 10001
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health/ready', timeout=3)"
CMD ["uvicorn", "localdesk.api:application", "--factory", "--host", "0.0.0.0", "--port", "8765", "--no-access-log"]
