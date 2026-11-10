Bu projede, elektrikli araç (EA) şarj istasyonu ağlarında kısa vadeli talep tahmini ve dinamik fiyatlandırma optimizasyonu için derin öğrenme ve topluluk (ensemble) tabanlı hibrit modeller incelenmiştir. Proje Tübitak 2209 kapsamında desteklenmektedir
Projenin amacı elektrikli araç şarj ağlarında dinamik ortamlı yapılan bir sistemde LSTM-XGBoost modelinin diğer modellere göre performansını test etmektir.
Sonuçlarla ilgili detaylar aşağıda mevcuttur.
Çalışmada kullanılan Shenzhen EA şarj veri seti bileşenleri ve proje içerisindeki görevleri aşağıda özetlenmiştir:

| Dosya | İlgili Modül | Açıklama |
|---|---|---|
| `time.csv` | `src/data_loader.py` | Zaman damgaları ve takvim özelliklerinin çıkarımı |
| `occupancy.csv` | `src/data_loader.py`, `src/features.py` | Ana hedef değişken (istasyon doluluk miktarı) |
| `volume.csv` | `src/features.py`, `src/pricing.py` | Şarj hacmi özelliği ve gelir simülasyonu hesabı |
| `duration.csv` | `src/features.py` | Şarj süresi özelliği |
| `price.csv` | `src/features.py`, `src/pricing.py` | Geçmiş fiyat bilgisi ve sabit zaman dilimli (ToU) referans tarife |
| `information.csv` | `train_demand_model.py`, `visualize_spatial.py` | Bölge kapasite bilgileri ve mekansal analiz |
| `adj.csv` | `src/features.py` | Komşu bölgelerin ortalama doluluk hesabı (komşuluk matrisi) |
| `distance.csv` | `config.py` | Bölgeler arası mesafe matrisi |
| `stations.csv` | `visualize_spatial.py` | Fiziksel şarj istasyonlarının konum bilgileri |
| `SZweather20220619-20220718.xls` | `src/data_loader.py`, `src/features.py` | Sıcaklık, nem ve rüzgar gibi dışsal meteorolojik değişkenler |


Gerekli Python paketlerinin kurulumu:

```bash
pip install pandas numpy scipy scikit-learn matplotlib tensorflow-cpu xgboost xlrd openpyxl
```

Veri setine ait `.csv` ve `.xls` dosyaları `data/` klasörüne yerleştirildikten sonra eğitim ve görselleştirme betikleri çalıştırılabilir:

```bash
# Temel LSTM, CNN-BiLSTM ve Attention modelleri
python train_demand_model.py --grid 102 --epochs 50
python train_demand_model.py --grid 102 --models CNN_BiLSTM_Attention,LSTM_baseline

# GRU hibritlerinin çoklu seed ile eğitimi
python train_gru_models.py --grid 102 --epochs 50 --seeds 42,43,44

# LSTM+XGBoost ve TCN+LSTM hibrit karşılaştırması
python train_hybrid_models.py --grid 102 --epochs 50 --seeds 42,43,44
python train_hybrid_models.py --grid 102 --target level
python train_hybrid_models.py --grid 102 --plots-only
```
5.1 CNN-LSTM ve Attention Modelleri
Grid 102 üzerinde tam eğitim (`--epochs 50`, EarlyStopping `patience=10`, `seed=42`) test sonuçları:

| Model | MAE | RMSE | MAPE (%) | R² |
|---|---|---|---|---|
| LSTM (baseline) | 0.442 | 0.574 | 2.47 | 0.963 |
| CNN-BiLSTM | 0.657 | 0.831 | 3.78 | 0.921 |
| CNN-BiLSTM-Attention | 0.570 | 0.769 | 3.18 | 0.933 |
| CNN-LSTM-Attention | 2.763 | 6.816 | 14.86 | -4.285 |

5.2. Hibrit Mimari Karşılaştırması (LSTM + XGBoost ve TCN + LSTM)
- **LSTM + XGBoost (Out-of-Fold Stacking):** Eğitim verisinde veri sızıntısını (data leakage) engellemek amacıyla zaman serisi 4 bloğa ayrılmış ve her bloğun LSTM tahmini, ilgili bloğu görmemiş modellerle üretilmiştir. Bu tahminler gecikmeli (lag) ve pencere özet istatistikleriyle birleştirilerek XGBoost'a aktarılmıştır.
- **TCN + LSTM:** Dilatasyon oranları `1, 2, 4, 8, 16` olan 5 nedensel (causal) residual TCN bloğu (etkin alıcı alan: 125 adım) ile yerel ve orta ölçekli örüntüler çıkarılıp LSTM katmanına verilmiştir.
- - **Fark (Delta) Hedef Dönüşümü:** 5 dakikalık çözünürlükte ardışık test adımlarının %84'ünde doluluk değişmemektedir. Doğrudan mutlak seviyeyi (`y(t+1)`) tahmin eden ilk deneylerde (v1, `outputs/hybrid/`) durağan adımlarda oluşan küçük sapmalar toplam hatayı yükseltmiştir.

Seviye hedefi (v1) ve fark hedefi + OOF stacking (v2) yaklaşımlarının ortalama MAE karşılaştırması:

| Model | v1 MAE (Seviye) | v2 MAE (Delta + OOF) | Değişim |
|---|---|---|---|
| Persistence | 0.164 | 0.164 | - |
| LSTM_baseline | 0.455 | 0.235 | -%48 |
| XGBoost_only | 0.249 | 0.204 | -%18 |
| LSTM_XGBoost | 0.332 | 0.201 | -%39 |
| TCN_only | 1.474 | 0.205 | -%86 |
| TCN_LSTM | 0.370 | 0.211 | -%43 |

Grid 102 üzerinde v2 düzeni (`--epochs 50`, 3 seed ortalaması ± standart sapma) ayrıntılı test sonuçları:

| Model | MAE | RMSE | MAPE (%) | R² | NN Parametre | Ağaç Sayısı | Süre (s) |
|---|---|---|---|---|---|---|---|
| Persistence | 0.1639 ± 0.0000 | 0.4224 ± 0.0000 | 0.950 ± 0.000 | 0.9797 ± 0.0000 | 0 | - | 0 |
| LSTM_baseline | 0.2351 ± 0.0185 | 0.4294 ± 0.0068 | 1.363 ± 0.099 | 0.9790 ± 0.0007 | 24,897 | - | 62 |
| XGBoost_only | 0.2039 ± 0.0026 | 0.4203 ± 0.0014 | 1.178 ± 0.016 | 0.9799 ± 0.0001 | 0 | 45 | 1 |
| LSTM_XGBoost | 0.2013 ± 0.0142 | 0.4200 ± 0.0026 | 1.162 ± 0.077 | 0.9799 ± 0.0002 | 24,897 | 38 | 236 |
| TCN_only | 0.2046 ± 0.0108 | 0.4230 ± 0.0015 | 1.174 ± 0.056 | 0.9796 ± 0.0001 | 32,801 | - | 154 |
| TCN_LSTM | 0.2110 ± 0.0198 | 0.4267 ± 0.0071 | 1.212 ± 0.106 | 0.9793 ± 0.0007 | 58,657 | - | 192 |

Modeller arası MAE farkları ve seed bazında Diebold-Mariano testi sonuçları:

| Karşılaştırma | Ortalama MAE Farkı | Seed 42 | Seed 43 | Seed 44 | Genel Değerlendirme |
|---|---|---|---|---|---|
| LSTM_XGBoost vs LSTM_baseline | %14.4 daha düşük | p < 0.001 (hibrit) | p < 0.001 (hibrit) | p = 0.09 | Hibrit üstün |
| LSTM_XGBoost vs XGBoost_only | %1.3 daha düşük | p = 0.40 | p < 0.001 (hibrit) | p = 0.001 (XGBoost) | Benzer performans |
| TCN_LSTM vs LSTM_baseline | %10.3 daha düşük | p < 0.001 (hibrit) | p < 0.001 (hibrit) | p = 0.02 (hibrit) | Hibrit üstün |
| TCN_LSTM vs TCN_only | %3.1 daha yüksek | p < 0.001 (TCN) | p < 0.001 (hibrit) | p = 0.33 | Benzer performans |
| LSTM_XGBoost vs TCN_LSTM | %4.6 daha düşük | p < 0.001 (LSTM_XGB) | p = 0.08 | p = 0.93 | LSTM_XGBoost önde |

Durağan ve değişim gösteren zaman adımlarına göre MAE dağılımı (3 seed ortalaması):

| Model | Değişmeyen Adımlar (%84) | Değişen Adımlar (%16) | Genel MAE |
|---|---|---|---|
| Persistence | 0.000 | 1.046 | 0.164 |
| LSTM_baseline | 0.090 | 1.018 | 0.235 |
| XGBoost_only | 0.052 | 1.020 | 0.204 |
| LSTM_XGBoost | 0.049 | 1.020 | 0.201 |
| TCN_only | 0.050 | 1.034 | 0.205 |
| TCN_LSTM | 0.059 | 1.030 | 0.211 |


Eğitilen modeller arasında en düşük MAE (0.2013) ve tüm modeller dahil en iyi RMSE (0.4200) ile R² (0.9799) değerleri `LSTM_XGBoost` hibrit modelinde elde edilmiştir. Out-of-fold kurgusu sayesinde LSTM öngörüsü, XGBoost karar ağaçlarında toplam kazanç (gain) açısından en yüksek katkıyı sağlayan ikinci özellik (%7.2 pay) olmuştur. Doluluğun değiştiği geçiş anlarında öğrenen modellerin tamamı naif `Persistence` yaklaşımından daha düşük hata üretirken, `LSTM_XGBoost` tahminleriyle yürütülen dinamik fiyatlandırma simülasyonunda gelir korunarak (%0.14 fark) zirve saatlerdeki talep yoğunluğu %3.1 oranında düşürülmüştür.
