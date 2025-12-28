# AestheticEnhancerAI 🎨✨

AI destekli görüntü iyileştirme ve restorasyon uygulaması. Eski, bozuk veya düşük kaliteli fotoğrafları yapay zeka kullanarak restore eder ve estetik açıdan geliştirir.

## 🌟 Özellikler

- **AI Yüz Restorasyonu**: GFPGAN ile yüz detaylarını iyileştirme
- **Süper Çözünürlük**: RealESRGAN ile 4x kalite artışı
- **Otomatik 2x Büyütme**: Tüm görseller otomatik olarak 2 kat büyütülür
- **Akıllı Temizlik**: Opsiyonel Gaussian blur ile gürültü giderme
- **Semantik Analiz**: Derinlik haritası ve nesne segmentasyonu
- **FastAPI Backend**: Mobil uygulama entegrasyonu için REST API
- **CLI Desteği**: Komut satırı arayüzü (geriye uyumlu)

## 📋 Gereksinimler

- Python 3.10+
- CUDA destekli GPU (opsiyonel, CPU'da da çalışır)
- 4GB+ RAM
- ~500MB disk alanı (model dosyaları için)

## 🚀 Kurulum

### 1. Repoyu klonlayın
```bash
git clone <repo-url>
cd AestheticEnhancerAI
```

### 2. Virtual environment oluşturun
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
# veya
venv\\Scripts\\activate  # Windows
```

### 3. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### 4. Model dosyalarını indirin
Model dosyaları otomatik olarak indirilir, ancak manuel indirmek için:

- **RealESRGAN**: [İndir](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth)
- **GFPGAN**: [İndir](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth)

İndirdiğiniz `.pth` dosyalarını proje kök dizinine koyun.

### 5. Environment dosyası (opsiyonel)
```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

## 💻 Kullanım

### FastAPI Backend (Mobil Uygulama için)

#### Sunucuyu başlatın:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### API Dokümantasyonu:
Sunucu başladıktan sonra tarayıcıdan:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

#### Endpoints:

**POST /api/v1/enhance** - Görüntü İyileştirme
```bash
curl -X POST "http://localhost:8000/api/v1/enhance" \\
  -F "image=@test_image.jpg" \\
  -F "denoise=false"
```

**GET /api/v1/health** - Health Check
```bash
curl "http://localhost:8000/api/v1/health"
```

#### Response Formatı:
```json
{
  "success": true,
  "enhanced_image": "base64_encoded_image...",
  "segmentation_map": "base64_encoded_map...",
  "original_size": {"width": 800, "height": 600},
  "output_size": {"width": 1600, "height": 1200},
  "processing_time": 45.3
}
```

### CLI Modu (Orijinal)

```bash
python main.py
```

1. Dosya seçici açılır
2. İşlenecek fotoğrafı seçin
3. Temizlik seçeneğini belirtin (evet/hayır)
4. Sonuçlar orijinal dosyanın yanına kaydedilir

## 📱 Mobil Uygulama Entegrasyonu

### React Native Örneği

```javascript
import { launchImageLibrary } from 'react-native-image-picker';

const enhanceImage = async () => {
  // Görsel seç
  const result = await launchImageLibrary({ mediaType: 'photo' });
  if (!result.assets?.[0]) return;
  
  const image = result.assets[0];
  
  // FormData oluştur
  const formData = new FormData();
  formData.append('image', {
    uri: image.uri,
    type: image.type,
    name: image.fileName,
  });
  formData.append('denoise', 'false');
  
  // API'ye gönder
  try {
    const response = await fetch('http://YOUR_SERVER_IP:8000/api/v1/enhance', {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    const data = await response.json();
    
    if (data.success) {
      // Base64'ü göster
      const enhancedImageUri = `data:image/jpeg;base64,${data.enhanced_image}`;
      setEnhancedImage(enhancedImageUri);
    }
  } catch (error) {
    console.error('Enhancement error:', error);
  }
};
```

### Flutter Örneği

```dart
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'dart:convert';

Future<void> enhanceImage() async {
  // Görsel seç
  final ImagePicker picker = ImagePicker();
  final XFile? image = await picker.pickImage(source: ImageSource.gallery);
  if (image == null) return;
  
  // Multipart request oluştur
  var request = http.MultipartRequest(
    'POST',
    Uri.parse('http://YOUR_SERVER_IP:8000/api/v1/enhance'),
  );
  
  request.files.add(await http.MultipartFile.fromPath('image', image.path));
  request.fields['denoise'] = 'false';
  
  // Gönder
  var response = await request.send();
  var responseData = await response.stream.bytesToString();
  var jsonData = json.decode(responseData);
  
  if (jsonData['success']) {
    String base64Image = jsonData['enhanced_image'];
    // Base64'ü göster
    setState(() {
      enhancedImage = base64Decode(base64Image);
    });
  }
}
```

## 🏗️ Proje Yapısı

```
AestheticEnhancerAI/
├── api/                    # FastAPI backend
│   ├── __init__.py
│   ├── main.py            # FastAPI app
│   ├── routes.py          # API endpoints
│   └── models.py          # Pydantic schemas
├── core/                  # İşleme mantığı
│   ├── __init__.py
│   ├── processor.py       # ImageProcessor class
│   └── utils.py           # Yardımcı fonksiyonlar
├── gfpgan/               # GFPGAN model dosyaları
├── main.py               # CLI versiyonu
├── requirements.txt      # Python bağımlılıkları
├── .env.example         # Environment şablonu
├── .gitignore          # Git ignore kuralları
└── README.md           # Bu dosya
```

## 🔧 Yapılandırma

`.env` dosyasında şu ayarları yapabilirsiniz:

```env
# Server
HOST=0.0.0.0
PORT=8000

# File Upload
MAX_FILE_SIZE=10485760  # 10 MB
ALLOWED_EXTENSIONS=jpg,jpeg,png,bmp

# AI Model
MODEL_DEVICE=cuda  # cuda veya cpu

# CORS
ALLOWED_ORIGINS=*  # Production'da specific origins kullanın
```

## 📊 İşleme Pipeline

1. **Temizlik** (Opsiyonel): Gaussian blur ile gürültü giderme
2. **AI Restorasyon**: GFPGAN + RealESRGAN ile iyileştirme
3. **Analiz**: Derinlik haritası + semantik segmentasyon
4. **Kompozisyon**: Akıllı birleştirme
5. **Efektler**: Master curve + Lightroom texture
6. **Büyütme**: Otomatik 2x upscaling

## ⚡ Performans

- **GPU (CUDA)**: ~15-30 saniye
- **CPU**: ~30-60 saniye

İşleme süresi görüntü boyutuna göre değişir.

## 🐛 Sorun Giderme

### Model dosyası bulunamadı
```bash
# Model dosyalarını manuel indirin ve proje dizinine koyun
```

### CUDA out of memory
```bash
# .env dosyasında MODEL_DEVICE=cpu yapın
```

### Basicsr import hatası
```bash
# Uygulama otomatik olarak düzeltir, ancak sorun devam ederse:
pip uninstall basicsr
pip install basicsr==1.4.2
```

## 📝 Lisans

MIT

## 🤝 Katkıda Bulunma

Pull request'ler kabul edilir. Büyük değişiklikler için önce issue açın.

## 📧 İletişim

Sorularınız için issue açabilirsiniz.

---

**Not**: Production ortamında:
- CORS ayarlarını sıkılaştırın
- API key authentication ekleyin  
- Rate limiting uygulayın
- HTTPS kullanın
