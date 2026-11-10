
import os

# ----------------------------------------------------------------------
# 1) VERİ YOLLARI
# ----------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

PATHS = {
    "time": os.path.join(DATA_DIR, "time.csv"),
    "occupancy": os.path.join(DATA_DIR, "occupancy.csv"),
    "volume": os.path.join(DATA_DIR, "volume.csv"),
    "duration": os.path.join(DATA_DIR, "duration.csv"),
    "price": os.path.join(DATA_DIR, "price.csv"),
    "information": os.path.join(DATA_DIR, "information.csv"),
    "adjacency": os.path.join(DATA_DIR, "adj.csv"),
    "distance": os.path.join(DATA_DIR, "distance.csv"),
    "stations": os.path.join(DATA_DIR, "stations.csv"),
    "weather": os.path.join(DATA_DIR, "SZweather20220619-20220718.xls"),
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 2) VERİ SETİ ÖZELLİKLERİ 
# ----------------------------------------------------------------------
TIMESTEP_MINUTES = 5          
STEPS_PER_DAY = 24 * 60 // TIMESTEP_MINUTES   # 288

# ----------------------------------------------------------------------
# 3) MODELLEME HİPERPARAMETRELERİ
# ----------------------------------------------------------------------
WINDOW_SIZE = 48        # 48 adım × 5 dk = 240 dk (4 saat) bağlam → CNN'e yeterli temporal derinlik
HORIZON = 1                # Bir sonraki adımı tahmin et. 
TARGET_COLUMN = "occupancy"  # occupancy | volume | duration  -> tahmin edilecek talep göstergesi

# Kronolojik train/val/test bölünmesi (gün bazında, TOPLAM 30 gün)
TRAIN_DAYS = 22
VAL_DAYS = 4
TEST_DAYS = 4            

BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3 #0.001 

# Hızlı test/örnek çalıştırma için (tam veriseti yerine belirli bir değer girdirmek için kullanılır.)
DEMO_MODE = False           
N_DEMO_GRIDS = 20
