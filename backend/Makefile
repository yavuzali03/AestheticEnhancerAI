SHELL := /bin/bash

.PHONY: start dev stop install clean

# Backend'i başlat
start:
	@echo "🚀 Backend başlatılıyor..."
	@source ../venv/bin/activate && uvicorn api.main:app --host 0.0.0.0 --port 8000

# Development mode (hot reload)
dev:
	@echo "🔥 Backend başlatılıyor (hot reload)..."
	@source ../venv/bin/activate && uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Backend'i durdur
stop:
	@echo "🛑 Backend durduruluyor..."
	@pkill -f "uvicorn api.main:app" || echo "Backend zaten durmuş"

# Dependencies kur
install:
	@echo "📦 Dependencies kuruluyor..."
	@source ../venv/bin/activate && pip install -r requirements.txt

# Cache temizle
clean:
	@echo "🧹 Cache temizleniyor..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
