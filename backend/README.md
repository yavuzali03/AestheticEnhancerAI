# AestheticEnhancerAI Backend

## Hızlı Başlatma

```bash
cd backend
make start     # Backend'i başlat
make dev       # Hot reload ile başlat
make stop      # Backend'i durdur
make install   # Dependencies kur
make clean     # Cache temizle
```

## Manuel Başlatma

```bash
cd backend
source ../venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- Health Check: `http://localhost:8000/api/v1/health`
- Enhance Image: `http://localhost:8000/api/v1/enhance`
- API Docs: `http://localhost:8000/docs`
