import os
import io
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

# --- Ayarlar ---
MUSIQ_MODEL_URL = 'https://tfhub.dev/google/musiq/ava/1'
TEST_IMAGE_PATH = 'rido.jpg'  # Analiz edilecek görsel
MODEL_REQUIRED_SIZE = (224, 224)


# --- "AŞÇININ TARİF DEFTERİ" (Tüm Tarifler) ---

def recipe_recover_highlights(pil_image):
    """TARİF 1: "Aşırı Pozlama" sorununu katmanlama tekniğiyle çözer."""
    print("     -> Tarif uygulanıyor: Patlamış Alanları Kurtarma...")
    # ... (kod aynı)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    factor = 0.80
    corrected_pil = ImageEnhance.Brightness(pil_image).enhance(factor)
    corrected_pil = ImageEnhance.Contrast(corrected_pil).enhance(1.1)
    corrected_cv = cv2.cvtColor(np.array(corrected_pil), cv2.COLOR_RGB2BGR)
    gray_original = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_original, 200, 255, cv2.THRESH_BINARY)
    mask_float = mask.astype(np.float32) / 255.0
    mask_3ch = cv2.merge([mask_float, mask_float, mask_float])
    img1 = cv_image.astype(np.float32)
    img2 = corrected_cv.astype(np.float32)
    blended_float = img1 * (1.0 - mask_3ch) + img2 * mask_3ch
    final_cv_image = np.uint8(np.clip(blended_float, 0, 255))
    return Image.fromarray(cv2.cvtColor(final_cv_image, cv2.COLOR_BGR2RGB))


def recipe_clahe_contrast_enhancement(pil_image):
    """TARİF 2 (SAYISAL): Düşük kontrastlı fotoğrafları CLAHE tekniği ile iyileştirir."""
    print("     -> Tarif uygulanıyor: CLAHE ile Adaptif Kontrast...")
    # ... (kod aynı)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(cv_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged_channels = cv2.merge([cl, a_channel, b_channel])
    final_cv_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2RGB)
    return Image.fromarray(final_cv_image)


def recipe_intelligent_crop(pil_image):
    """TARİF 3 (YAPAY ZEKA DESTEKLİ): Kompozisyonu iyileştirmek için akıllı kırpma (re-framing) yapar."""
    print("     -> Tarif uygulanıyor: Kompozisyon için Akıllı Kırpma...")
    # ... (kod aynı)
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print(f"     ❌ HATA: '{cascade_path}' bulunamadı. Lütfen indirip proje klasörüne koyun.")
        return pil_image
    face_cascade = cv2.CascadeClassifier(cascade_path)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    if len(faces) == 0:
        print("     -> Bilgi: Fotoğrafta yüz tespit edilemedi. Kırpma işlemi atlanıyor.")
        return pil_image
    height, width = pil_image.size[1], pil_image.size[0]
    interest_center_x = int(np.mean([x + w / 2 for x, y, w, h in faces]))
    interest_center_y = int(np.mean([y + h / 2 for x, y, w, h in faces]))
    power_points = [(width // 3, height // 3), (2 * width // 3, height // 3), (width // 3, 2 * height // 3),
                    (2 * width // 3, 2 * height // 3)]
    closest_point = min(power_points, key=lambda p: (p[0] - interest_center_x) ** 2 + (p[1] - interest_center_y) ** 2)
    dx = closest_point[0] - interest_center_x
    dy = closest_point[1] - interest_center_y
    new_x1 = max(0, 0 + dx)
    new_y1 = max(0, 0 + dy)
    new_x2 = min(width, width + dx)
    new_y2 = min(height, height + dy)
    crop_width = new_x2 - new_x1
    crop_height = new_y2 - new_y1
    if crop_width / crop_height > width / height:
        new_y1 = int(new_y1 - ((crop_width * height / width) - crop_height) / 2)
        new_y2 = new_y1 + int(crop_width * height / width)
    else:
        new_x1 = int(new_x1 - ((crop_height * width / height) - crop_width) / 2)
        new_x2 = new_x1 + int(crop_height * width / height)
    final_x1, final_y1, final_x2, final_y2 = max(0, new_x1), max(0, new_y1), min(width, new_x2), min(height, new_y2)
    return pil_image.crop((final_x1, final_y1, final_x2, final_y2))


def recipe_shadow_recovery(pil_image):
    """TARİF 4: Fotoğrafın karanlık bölgelerindeki (gölgelerdeki) detayları ortaya çıkarır."""
    print("     -> Tarif uygulanıyor: Gölgeleri Kurtarma...")
    # ... (kod aynı)
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    l_gamma_corrected = cv2.LUT(l_channel, table)
    merged_channels = cv2.merge([l_gamma_corrected, a_channel, b_channel])
    final_cv_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(final_cv_image, cv2.COLOR_BGR2RGB))


def recipe_vibrance_and_saturation(pil_image):
    """TARİF 5 (HASSAS): Renkleri doğal bir şekilde canlandırır."""
    print("     -> Tarif uygulanıyor: Renk Canlılığını Artırma...")
    converter = ImageEnhance.Color(pil_image)
    enhanced_image = converter.enhance(1.45) # Daha belirgin bir etki için artırıldı
    return enhanced_image


def recipe_unsharp_mask(pil_image):
    """TARİF 6 (HASSAS): Görseli "Unsharp Mask" ile akıllıca keskinleştirir."""
    print("     -> Tarif uygulanıyor: Akıllı Keskinleştirme (Unsharp Mask)...")
    return pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


# --- "DANIŞMAN MODÜLÜ" (Tüm Danışmanlar) ---

def analyze_exposure(pil_image):
    """DANIŞMAN 1: Fotoğrafın pozlamasını analiz eder."""
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(cv_image)
    if mean_brightness < 70:
        return "DÜŞÜK POZLAMA", f"Fotoğraf çok karanlık (Ort. Parlaklık: {mean_brightness:.0f})."
    elif mean_brightness > 185:
        return "YÜKSEK POZLAMA", f"Fotoğraf çok parlak (Ort. Parlaklık: {mean_brightness:.0f})."
    else:
        return "İYİ POZLAMA", f"Pozlama dengeli (Ort. Parlaklık: {mean_brightness:.0f})."


def analyze_contrast_with_histogram(pil_image):
    """DANIŞMAN 2 (SAYISAL): Histogram analizi ile kontrastı ölçer."""
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([cv_image], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    p5 = np.searchsorted(cdf, cdf[-1] * 0.05)
    p95 = np.searchsorted(cdf, cdf[-1] * 0.95)
    dynamic_range = p95 - p5
    if dynamic_range < 100:
        return "DÜŞÜK KONTRAST", f"Pikseller dar bir aralığa sıkışmış (Dinamik Aralık: {dynamic_range}). Fotoğraf puslu/soluk."
    else:
        return "İYİ KONTRAST", f"Kontrast seviyesi dengeli (Dinamik Aralık: {dynamic_range})."


def analyze_rule_of_thirds(pil_image):
    """DANIŞMAN 3: Kompozisyonu, yüz tespiti yaparak analiz eder."""
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        return "ANALİZ EDİLEMEDİ", "Haar Cascade dosyası eksik."
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)
    if len(faces) == 0:
        return "BELİRSİZ", "Odak noktası (yüz) tespit edilemedi."
    height, width = pil_image.size[1], pil_image.size[0]
    interest_center_x = int(np.mean([x + w / 2 for x, y, w, h in faces]))
    interest_center_y = int(np.mean([y + h / 2 for x, y, w, h in faces]))
    center_threshold_x = width * 0.2
    center_threshold_y = height * 0.2
    if (width / 2 - center_threshold_x < interest_center_x < width / 2 + center_threshold_x) and \
            (height / 2 - center_threshold_y < interest_center_y < height / 2 + center_threshold_y):
        return "MERKEZİ KOMPOZİSYON", "Ana obje (yüzler) kadrajın merkezinde."
    else:
        return "DENGELİ KOMPOZİSYON", "Ana obje (yüzler) merkez dışında."


def analyze_color_vibrance(pil_image):
    """DANIŞMAN 4 (HASSAS): Renklerin canlılığını (vibrance) analiz eder."""
    hsv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2HSV)
    saturation_channel = hsv_image[:, :, 1]
    mean_saturation = np.mean(saturation_channel)
    if mean_saturation < 80:
        return "DÜŞÜK CANLILIK", f"Renkler soluk görünüyor (Ort. Doygunluk: {mean_saturation:.0f})."
    else:
        return "İYİ CANLILIK", f"Renkler yeterince canlı (Ort. Doygunluk: {mean_saturation:.0f})."


def analyze_sharpness(pil_image):
    """DANIŞMAN 5 (HASSAS): Görselin netliğini/keskinliğini analiz eder."""
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    variance = cv2.Laplacian(cv_image, cv2.CV_64F).var()
    if variance < 100:
        return "YUMUŞAK ODAK", f"Görsel biraz yumuşak (Netlik Skoru: {variance:.0f})."
    else:
        return "İYİ KESKİNLİK", f"Görsel yeterince keskin (Netlik Skoru: {variance:.0f})."


# --- ÇEKİRDEK YARDIMCI FONKSİYONLAR ---

def aspect_ratio_pad_resize(pil_image, target_size):
    """Görselin en-boy oranını koruyarak hedef boyuta sığdırır ve dolgu ekler."""
    # ... (kod aynı)
    original_width, original_height = pil_image.size
    target_width, target_height = target_size
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)
    resized_img = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    delta_w, delta_h = target_width - new_width, target_height - new_height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    return ImageOps.expand(resized_img, padding, fill=(128, 128, 128))


def get_score_for_pil(pil_image, model_predictor):
    """Verilen bir PIL görüntüsünü, en-boy oranını koruyarak skorlar."""
    # ... (kod aynı)
    processed_image = aspect_ratio_pad_resize(pil_image, MODEL_REQUIRED_SIZE)
    with io.BytesIO() as buffer:
        img_format = 'PNG' if pil_image.format in ['PNG', None] else 'JPEG'
        processed_image.save(buffer, format=img_format)
        image_bytes = buffer.getvalue()
    image_tensor = tf.constant(image_bytes, dtype=tf.string)
    inputs = {'image_bytes_tensor': image_tensor}
    try:
        score_tensor = model_predictor(**inputs)
        return score_tensor['output_0'].numpy()
    except Exception as e:
        print(f"❌ Skorlama sırasında hata: {e}")
        return 0.0


# --- AKILLI OPTİMİZASYON MOTORU (GÜNCELLENMİŞ EYLEM PLANI) ---

# --- AKILLI OPTİMİZASYON MOTORU (GÜNCELLENMİŞ EYLEM PLANI) ---

# --- AKILLI OPTİMİZASYON MOTORU (GÜNCELLENMİŞ EYLEM PLANI) ---

def optimize_for_score(original_pil_image, initial_score, model_predictor, analysis_report_statuses):
    """Tespit edilen temel ve hassas sorunları çözmeye yönelik tarifleri dener."""
    print("\n" + "=" * 50)
    print("🤖 Akıllı Optimizasyon Motoru Başlatılıyor...")
    best_image, best_score, action_taken = original_pil_image, initial_score, False

    # Tariflerin sırası önemli: önce temel sorunlar, sonra hassas iyileştirmeler.
    # Kompozisyon en son olmalı.
    potential_recipes = [
        # Temel Sorun Gidericiler
        ("YÜSEK POZLAMA", "Patlamış Alanları Kurtarma", recipe_recover_highlights),
        ("DÜŞÜK KONTRAST", "CLAHE ile Adaptif Kontrast", recipe_clahe_contrast_enhancement),

        # Hassas İyileştiriciler (Analiz raporunda tespit edilirse veya 'İYİ' olsa bile denenmesi istenirse)
        # Bu kısımda "HER ZAMAN_" yerine daha spesifik koşullar veya mantık kullanalım.
        ("DÜŞÜK CANLILIK", "Renk Canlılığını Artırma", recipe_vibrance_and_saturation),  # Raporda DÜŞÜK CANLILIK varsa
        ("İYİ CANLILIK_GELISTIR", "Renk Canlılığını Artırma (Daha da)", recipe_vibrance_and_saturation),
        # İYİ olsa bile geliştirilebilir olarak dene
        ("YUMUŞAK ODAK", "Akıllı Keskinleştirme", recipe_unsharp_mask),  # Raporda YUMUŞAK ODAK varsa
        ("İYİ KESKİNLIK_GELISTIR", "Akıllı Keskinleştirme (Daha da)", recipe_unsharp_mask),
        # İYİ olsa bile geliştirilebilir olarak dene
        ("İYİ POZLAMA_GOLGE", "Gölgeleri Kurtarma", recipe_shadow_recovery),  # Pozlama iyi olsa bile gölgeleri kurtar
        ("DÜŞÜK POZLAMA_GOLGE", "Gölgeleri Kurtarma", recipe_shadow_recovery),
        # Pozlama düşükse de gölgeleri kurtar (öncelik ver)

        # Kompozisyon (En Son)
        ("MERKEZİ KOMPOZİSYON", "Kompozisyon için Akıllı Kırpma", recipe_intelligent_crop),
    ]

    for condition, name, operation in potential_recipes:
        should_apply_recipe = False

        if condition in analysis_report_statuses:  # Eğer rapor direk bu condition'ı içeriyorsa
            should_apply_recipe = True
        elif condition == "İYİ CANLILIK_GELISTIR" and "İYİ CANLILIK" in analysis_report_statuses:
            should_apply_recipe = True
        elif condition == "İYİ KESKİNLIK_GELISTIR" and "İYİ KESKİNLİK" in analysis_report_statuses:
            should_apply_recipe = True
        elif condition == "İYİ POZLAMA_GOLGE" and "İYİ POZLAMA" in analysis_report_statuses:
            should_apply_recipe = True
        elif condition == "DÜŞÜK POZLAMA_GOLGE" and "DÜŞÜK POZLAMA" in analysis_report_statuses:
            should_apply_recipe = True

        if should_apply_recipe:
            action_taken = True
            print(f"\n   - Durum: '{condition}'. Çözüm deneniyor: '{name}'...")

            candidate_image = operation(best_image)

            if name == "Kompozisyon için Akıllı Kırpma" and \
                    (
                            candidate_image.width < best_image.width * 0.7 or candidate_image.height < best_image.height * 0.7):
                print("     ❌ Başarısız. Kırpma sonucu çok küçük, değişiklik geri alınıyor.")
                continue

            new_score = get_score_for_pil(candidate_image, model_predictor)
            if new_score > best_score:
                print(f"     ✅ BAŞARILI! Skor {best_score:.2f} -> {new_score:.2f}'a yükseldi. Bu değişiklik kalıcı.")
                best_image, best_score = candidate_image, new_score
            else:
                print(f"     ❌ Başarısız. Skor düştü veya aynı kaldı ({new_score:.2f}). Değişiklik geri alınıyor.")

    if not action_taken and initial_score == best_score:
        print("   - Analiz raporunda eyleme geçirilebilir bir sorun bulunamadı ve hiçbir tarif skoru artıramadı.")

    print(f"\n✅ Optimizasyon tamamlandı. Nihai Skor: {best_score:.2f}")
    return best_image


# --- ANA UYGULAMA (GÜNCELLENMİŞ RAPORLAMA) ---

def main():
    """Uygulamanın ana akışını yönetir."""
    print("--- Aesthetic Enhancer AI (Hassas İyileştirme Sürümü) Başlatılıyor ---")

    # Adım 1: Modeli yükle
    print("\n🧠 MUSIQ Modeli yükleniyor...")
    try:
        musiq_model = hub.load(MUSIQ_MODEL_URL)
        predictor = musiq_model.signatures["serving_default"]
        print("✅ Model başarıyla yüklendi.")
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {e}"); return

    # Adım 2: Görseli yükle ve başlangıç skorunu al
    try:
        original_pil_image = Image.open(TEST_IMAGE_PATH)
        # OTOMATİK ROTASYON DÜZELTME
        original_pil_image = ImageOps.exif_transpose(original_pil_image).convert("RGB")
        print("✅ Görsel başarıyla yüklendi ve EXIF rotasyonu düzeltildi.")
        initial_score = get_score_for_pil(original_pil_image, predictor)
    except FileNotFoundError:
        print(f"❌ HATA: '{TEST_IMAGE_PATH}' bulunamadı."); return
    except Exception as e:
        print(f"❌ Görsel yüklenirken hata: {e}"); return

    print("\n" + "=" * 50)
    print(f"🖼️ Analiz Edilen Görsel: {TEST_IMAGE_PATH}")
    print(f"✨ Başlangıç Estetik Skoru: {initial_score:.2f}")

    # Adım 3: Anlık Geri Bildirim Raporu Oluştur (Tüm Analizler)
    print("\n" + "-" * 50)
    print("🧐 Anlık Geri Bildirim Raporu (Sayısal ve Hassas Analiz):")
    analysis_report_statuses = []

    # Raporlama sırasını mantıklı hale getirelim
    status, suggestion = analyze_exposure(original_pil_image)
    print(f"   - Pozlama: {status} -> {suggestion}")
    analysis_report_statuses.append(status)

    status, suggestion = analyze_contrast_with_histogram(original_pil_image)
    print(f"   - Kontrast: {status} -> {suggestion}")
    analysis_report_statuses.append(status)

    status, suggestion = analyze_color_vibrance(original_pil_image)
    print(f"   - Renk Canlılığı: {status} -> {suggestion}")
    analysis_report_statuses.append(status)

    status, suggestion = analyze_sharpness(original_pil_image)
    print(f"   - Netlik: {status} -> {suggestion}")
    analysis_report_statuses.append(status)

    status, suggestion = analyze_rule_of_thirds(original_pil_image)
    print(f"   - Kompozisyon: {status} -> {suggestion}")
    analysis_report_statuses.append(status)
    print("-" * 50)

    # Adım 4: Akıllı Optimizasyon Motorunu Çalıştır
    enhanced_image = optimize_for_score(original_pil_image, initial_score, predictor, analysis_report_statuses)

    # Adım 5: Sonucu Kaydet
    if enhanced_image is not original_pil_image:
        save_path = "enhanced_" + os.path.basename(TEST_IMAGE_PATH)
        enhanced_image.save(save_path)
        print(f"\n💾 Optimize edilmiş görsel '{save_path}' adıyla kaydedildi.")
    else:
        print("\n💾 Görselde iyileştirme yapılmadığı veya bulunmadığı için yeni dosya kaydedilmedi.")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()