import os
import io
import cv2
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageStat

# --- Ayarlar ---
MUSIQ_MODEL_URL = 'https://tfhub.dev/google/musiq/ava/1'
TEST_IMAGE_PATH = 'test3.png'  # Analiz edilecek görsel
TARGET_SIZE = (512, 512)


# --- GÖRSEL İŞLEME VE PUANLAMA FONKSİYONLARI (Değişiklik yok) ---
def load_and_prepare_image_bytes(image_path):
    if not os.path.exists(image_path): print(f"❌ HATA: '{image_path}' dosyası bulunamadı."); return None
    try:
        print(f"📁 Orijinal görsel yükleniyor: {image_path}");
        img = Image.open(image_path).convert('RGB');
        print(f"📏 Orijinal Boyut: {img.size}");
        print(f"🔄 Görsel {TARGET_SIZE} boyutuna getiriliyor...");
        img_resized = img.resize(TARGET_SIZE)
        with io.BytesIO() as buffer:
            img_resized.save(buffer, format='PNG'); image_bytes = buffer.getvalue()
        return tf.constant(image_bytes, dtype=tf.string)
    except Exception as e:
        print(f"❌ HATA: Görsel yüklenirken veya işlenirken bir sorun oluştu: {e}"); return None


def calculate_aesthetic_score(image_tensor):
    print("\n🧠 MUSIQ Modeli yükleniyor...");
    try:
        musiq_model = hub.load(MUSIQ_MODEL_URL);
        print("✅ Model başarıyla yüklendi.");
        print("🔮 Estetik skor hesaplanıyor...")
        predictor = musiq_model.signatures["serving_default"];
        inputs = {'image_bytes_tensor': image_tensor}
        score_tensor = predictor(**inputs);
        score = score_tensor['output_0'].numpy()
        return score
    except Exception as e:
        print(f"❌ Model yüklenirken veya tahmin yaparken bir hata oluştu: {e}"); return None


# --- YENİ: RAPORLAMA ODAKLI ANALİZ FONKSİYONLARI ---

def analyze_brightness_and_contrast(pil_image):
    results = {}
    try:
        grayscale_img = pil_image.convert('L');
        stats = ImageStat.Stat(grayscale_img)
        # Parlaklık Analizi
        avg_brightness = stats.mean[0]
        brightness_status = "Dengeli"
        brightness_suggestion = None
        if avg_brightness < 70:
            brightness_status = "Karanlık"; brightness_suggestion = "Aydınlatma düşük. Işığı veya pozlamayı artırın."
        elif avg_brightness > 185:
            brightness_status = "Aşırı Aydınlık"; brightness_suggestion = "Aşırı aydınlık veya patlamış alanlar var. Pozlamayı azaltın."
        results['Parlaklık'] = {'value': f"{avg_brightness:.2f} / 255 ({brightness_status})",
                                'suggestion': brightness_suggestion}

        # Kontrast Analizi
        std_dev = stats.stddev[0]
        contrast_status = "İyi"
        contrast_suggestion = None
        if std_dev < 40: contrast_status = "Düşük"; contrast_suggestion = "Kontrast düşük. Daha belirgin gölgeler ve ışıklar için kontrastı artırın."
        results['Kontrast'] = {'value': f"{std_dev:.2f} (StdDev)", 'suggestion': contrast_suggestion}

    except Exception as e:
        results['Genel Hata'] = {'value': str(e), 'suggestion': 'Analiz sırasında bir hata oluştu.'}
    return results


def analyze_saturation(pil_image):
    value_str, suggestion = "N/A", None
    try:
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR);
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        saturation_channel = hsv_image[:, :, 1];
        value_channel = hsv_image[:, :, 2]
        brightness_mask = (value_channel > 50) & (value_channel < 220);
        color_mask = saturation_channel > 30
        final_mask = brightness_mask & color_mask;
        meaningful_saturations = saturation_channel[final_mask]

        if meaningful_saturations.size > 0:
            avg_saturation = np.mean(meaningful_saturations)
            status = "Dengeli"
            if avg_saturation < 60:
                status = "Soluk"; suggestion = "Renkler soluk. Doygunluğu artırın."
            elif avg_saturation > 190:
                status = "Aşırı Canlı"; suggestion = "Renkler aşırı canlı. Doygunluğu azaltarak doğallaştırın."
            value_str = f"{avg_saturation:.2f} / 255 ({status})"
        else:
            value_str = "Anlamlı renk bulunamadı."

    except Exception as e:
        value_str = str(e); suggestion = 'Analiz sırasında bir hata oluştu.'
    return {'Doygunluk': {'value': value_str, 'suggestion': suggestion}}


def analyze_color_balance(pil_image):
    value_str, suggestion = "N/A", None
    try:
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR);
        lab_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        mean_b = np.mean(lab_image[:, :, 2]);
        status = "Dengeli"
        if mean_b < 120:
            status = "Soğuk"; suggestion = "Renk dengesi soğuk tonlarda. Daha sıcak bir filtre veya beyaz ayarı deneyin."
        elif mean_b > 135:
            status = "Sıcak"; suggestion = "Renk dengesi sıcak tonlarda. Daha soğuk bir filtre veya beyaz ayarı deneyin."
        value_str = f"{mean_b:.2f} (Lab 'b' kanalı - Nötr: ~128) ({status})"

    except Exception as e:
        value_str = str(e); suggestion = 'Analiz sırasında bir hata oluştu.'
    return {'Renk Dengesi': {'value': value_str, 'suggestion': suggestion}}


def analyze_composition(image_path):
    value_str, suggestion = "N/A", None
    try:
        image = cv2.imread(image_path);
        height, width, _ = image.shape
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        (success, saliency_map) = saliency.computeSaliency(image);
        _, _, _, max_loc = cv2.minMaxLoc(saliency_map)
        interest_point = np.array(max_loc)

        third_w, third_h = width / 3, height / 3
        rule_of_thirds_points = np.array(
            [(third_w, third_h), (2 * third_w, third_h), (third_w, 2 * third_h), (2 * third_w, 2 * third_h)])
        distances = np.linalg.norm(rule_of_thirds_points - interest_point, axis=1)
        closest_point_index = np.argmin(distances);
        closest_point = rule_of_thirds_points[closest_point_index];
        min_dist = distances[closest_point_index]
        diagonal = np.sqrt(width ** 2 + height ** 2);
        status = "İyi"

        if min_dist > diagonal * 0.15:
            status = "Zayıf"
            diff_vector = closest_point - interest_point
            if abs(diff_vector[1]) > abs(diff_vector[0]):
                if diff_vector[1] < 0:
                    suggestion = "Kompozisyon için kadrajı biraz yukarı alın."
                else:
                    suggestion = "Kompozisyon için kadrajı biraz aşağı alın."
            else:
                if diff_vector[0] < 0:
                    suggestion = "Kompozisyon için kadrajı biraz sola alın."
                else:
                    suggestion = "Kompozisyon için kadrajı biraz sağa alın."
        value_str = f"İlgi odağı güçlü noktaya {min_dist:.0f} piksel uzakta ({status})"

    except Exception as e:
        value_str = str(e); suggestion = 'Analiz sırasında bir hata oluştu.'
    return {'Kompozisyon': {'value': value_str, 'suggestion': suggestion}}


def get_all_analysis_results(image_path):
    print("\n" + "-" * 50);
    print("🔬 Gelişmiş Analizler Başlatılıyor...")
    pil_image = Image.open(image_path).convert('RGB')

    # Tüm analizleri çalıştır ve sonuçları tek bir sözlükte birleştir
    analysis_report = {}
    analysis_report.update(analyze_brightness_and_contrast(pil_image))
    analysis_report.update(analyze_saturation(pil_image))
    analysis_report.update(analyze_color_balance(pil_image))
    analysis_report.update(analyze_composition(image_path))
    return analysis_report


# --- ANA FONKSİYON ---
def main():
    print("--- Aesthetic Enhancer AI - Yerel Görsel Demo Başlatılıyor ---")
    image_tensor = load_and_prepare_image_bytes(TEST_IMAGE_PATH)
    if image_tensor is None: return
    aesthetic_score = calculate_aesthetic_score(image_tensor)
    if aesthetic_score is None: return

    formatted_score = f"{aesthetic_score:.2f}"
    print("\n" + "=" * 50);
    print(f"🖼️ Analiz Edilen Görsel: {TEST_IMAGE_PATH}");
    print(f"✨ Tahmini Estetik Skoru (1-10): {formatted_score}");
    print("=" * 50)

    # Analiz raporunu al
    analysis_report = get_all_analysis_results(TEST_IMAGE_PATH)

    # Raporu ve önerileri formatlı bir şekilde yazdır
    print("\n📋 TEKNİK ANALİZ RAPORU:")
    suggestions = []
    for analysis_name, result in analysis_report.items():
        print(f"  - {analysis_name:<15}: {result.get('value', 'N/A')}")
        if result.get('suggestion'):
            suggestions.append(result['suggestion'])

    print("\n" + "-" * 50)
    if suggestions:
        print("💡 İyileştirme Önerileri:")
        for i, suggestion in enumerate(suggestions, 1): print(f"   {i}. {suggestion}")
    else:
        print("💡 Fotoğrafınız temel teknik analizlere göre oldukça dengeli görünüyor!")
    print("-" * 50)

    if aesthetic_score > 7.5:
        print("🌟 Olağanüstü! Profesyonel seviyede estetik.")
    elif aesthetic_score > 5.5:
        print("👍 İyi bir fotoğraf. Estetik açıdan sağlam görünüyor.")
    else:
        print("⚠️ Geliştirilebilir.")
        if suggestions:
            print(
                "   Bu puan, yukarıda listelenen spesifik teknik ve kompozisyonel iyileştirme alanlarından kaynaklanıyor olabilir.")
        else:
            print(
                "   Fotoğrafınız temel teknik kurallara uygun görünse de, genel estetik etki (renk uyumu, ışık kalitesi, konu gibi daha soyut unsurlar) puanı etkilemiş olabilir.")


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()