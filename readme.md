# LSTM & LLM Support System Project

Bu proje, LSTM (Long Short-Term Memory) ağları ve LLM (Large Language Model) teknolojilerini bir araya getirerek geliştirilen bir destek sistemidir. Proje kapsamında hem zaman serisi/metin dizisi tahminlemeleri hem de gelişmiş dil işleme yetenekleri bir arada sunulmaktadır.

## 🚀 Proje İçeriği

- **LSTM Modeli:** Zaman serisi analizi veya yapılandırılmış metin verileri üzerinde tahminleme.
- **LLM Entegrasyonu:** Karmaşık sorguların işlenmesi ve anlamlandırılması için büyük dil modeli desteği.
- **Data Pipeline:** Verilerin işlenmesi ve modellerin eğitimi için optimize edilmiş iş akışı.

## 📁 Dosya Yapısı

- `test.py`: Temel test ve çalıştırma betiği.
- `requirements.txt`: Projenin çalışması için gerekli kütüphaneler.
- `.env`: API anahtarları ve hassas yapılandırmalar (Git'e dahil edilmez).
- `human_decision_log.jsonl`: Karar süreçlerine dair kayıt dosyaları.

## 🛠️ Kurulum

1. Depoyu yerel bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/mcozkan/LSTM_LLM_SupportSystem_project.git](https://github.com/mcozkan/LSTM_LLM_SupportSystem_project.git)

2. Sanal ortamı oluşturun ve aktif edin:
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    veya venv\Scripts\activate # Windows
     **NOTE** ben python3.12 -m venv venv şeklinde oluşturdum. 

3. Gerekli paketleri yükleyin:
    pip install -r requirements.txt

4. Kullanım:
    Uygulamayı yerel sunucuda başlatmak için terminale aşağıdaki komutu yazın:

    ```bash
    streamlit run test.py
