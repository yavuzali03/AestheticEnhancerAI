# AestheticEnhancerAI - Monorepo

AI destekli görüntü iyileştirme uygulaması. FastAPI backend ve React Native mobil uygulama içerir.

## 📁 Proje Yapısı

```
AestheticEnhancerAI/
├── backend/              # FastAPI Backend
│   ├── api/             # API endpoints
│   ├── core/            # İşleme mantığı
│   └── main.py          # CLI tool
├── mobile/              # React Native Mobile App
│   ├── src/
│   │   ├── screens/    # UI screens
│   │   ├── services/   # API services
│   │   └── config/     # Configuration
│   └── App.jsx
├── *.pth                # AI Model dosyaları
└── README.md           # Bu dosya
```

## 🚀 Hızlı Başlangıç

### Backend (FastAPI)

```bash
cd backend
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Dependencies (ilk seferinde)
pip install -r requirements.txt

# Sunucuyu başlat
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**API Dokümantasyon:** http://localhost:8000/docs

### Mobile App (React Native)

```bash
cd mobile

# Dependencies (ilk seferinde)
npm install
cd ios && pod install && cd ..  # iOS için

# iOS
npm run ios

# Android
npm run android
```

## 📱 Mobil Uygulama Özellikleri

- ✅ Galeriden görsel seçme
- ✅ Kameradan fotoğraf çekme
- ✅ Otomatik 2x büyütme
- ✅ Gürültü temizleme (denoise) seçeneği
- ✅ İşlem ilerleme göstergesi
- ✅ Before/After karşılaştırma
- ✅ Segmentasyon haritası görüntüleme
- ✅ Backend bağlantı durumu kontrolü

## 🔧 Backend API

### Endpoints

**POST /api/v1/enhance**
```bash
curl -X POST "http://localhost:8000/api/v1/enhance" \\
  -F "image=@photo.jpg" \\
  -F "denoise=false"
```

**Response:**
```json
{
  "success": true,
  "enhanced_image": "base64...",
  "segmentation_map": "base64...",
  "original_size": {"width": 800, "height": 600},
  "output_size": {"width": 1600, "height": 1200},
  "processing_time": 45.2
}
```

## 💻 Geliştirme

### Backend URL Yapılandırması

Mobile app varsayılan olarak şu URL'leri kullanır:
- **iOS Simulator:** `http://localhost:8000`
- **Android Emulator:** `http://10.0.2.2:8000`
- **Gerçek Cihaz:** Local IP'nizi `mobile/src/config/api.js` dosyasında güncelleyin

### Test Etme

1. Backend'i başlatın (port 8000)
2. Mobile app'i simulator'de çalıştırın
3. Galeriden görsel seçin
4. "İyileştir" butonuna basın
5. Sonucu görüntüleyin

## 📖 Detaylı Dokümantasyon

- **Backend:** [backend/README_BACKEND.md](backend/README_BACKEND.md)
- **Mobile:** [mobile/README.md](mobile/README.md)

## 🛠️ Teknoloji Stack

**Backend:**
- FastAPI
- PyTorch
- GFPGAN
- RealESRGAN
- Transformers

**Mobile:**
- React Native 0.80
- React Navigation
- Axios
- react-native-image-picker

## 📝 Lisans

MIT

## 🤝 Katkıda Bulunma

Pull request'ler kabul edilir. Büyük değişiklikler için önce issue açın.

---

**Not:** Backend ve mobile app ayrı ayrı çalıştırılmalıdır. Mobil app'in backend'e erişebilmesi için backend sunucusu çalışır durumda olmalıdır.
