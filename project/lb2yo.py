# Labelme futtatása: python -m labelme
import json
import os
import glob
import shutil
from sklearn.model_selection import train_test_split

# --- ALAP ÚTVONALAK ---
# Megkeressük, hol van ez a script
alap_utvonal = os.path.dirname(os.path.abspath(__file__))

# Bemeneti mappa (JSON + képek)
bemeneti_mappa = os.path.join(alap_utvonal, 'pct')

# Kimeneti mappa (ide készül a YOLO adat)
kimeneti_mappa = os.path.join(alap_utvonal, 'YOLOData')

# Osztályok
osztalyok = ["5ft", "10ft", "20ft", "50ft", "100ft", "200ft"]

# --- MAPPÁK LÉTREHOZÁSA ---
for resz in ['train', 'val']:
    os.makedirs(os.path.join(kimeneti_mappa, resz, 'images'), exist_ok=True)
    os.makedirs(os.path.join(kimeneti_mappa, resz, 'labels'), exist_ok=True)

# JSON fájlok keresése
json_fajlok = glob.glob(os.path.join(bemeneti_mappa, "*.json"))

if not json_fajlok:
    print(f"HIBA: Nem találtam JSON fájlokat itt: {bemeneti_mappa}")
    exit()

# Train / validation szétosztás
train_fajlok, val_fajlok = train_test_split(json_fajlok, test_size=0.2, random_state=42)

# --- KONVERTÁLÁS YOLO FORMÁTUMBA ---
def atalakit_yolo(fajlok, resz):
    for json_utvonal in fajlok:
        with open(json_utvonal, 'r', encoding='utf-8') as f:
            adat = json.load(f)

            kep_nev = adat['imagePath']
            kep_fajlnev = os.path.basename(kep_nev)
            #ezek kellenek a normalizáláshoz
            kep_szelesseg = adat['imageWidth']
            kep_magassag = adat['imageHeight']

            # Kép másolása
            forras_kep = os.path.join(bemeneti_mappa, kep_fajlnev)
            cel_kep = os.path.join(kimeneti_mappa, resz, 'images', kep_fajlnev)

            if os.path.exists(forras_kep):
                shutil.copy(forras_kep, cel_kep)
            else:
                print(f"FIGYELEM: Hiányzik a kép: {forras_kep}")

            # TXT fájl létrehozása
            txt_nev = os.path.splitext(kep_fajlnev)[0] + ".txt"
            txt_utvonal = os.path.join(kimeneti_mappa, resz, 'labels', txt_nev)

            with open(txt_utvonal, 'w') as ki:
                for alakzat in adat['shapes']:
                    cimke = alakzat['label']

                    if cimke not in osztalyok:
                        continue

                    osztaly_id = osztalyok.index(cimke)

                    pontok = alakzat['points']
                    x1, y1 = pontok[0]
                    x2, y2 = pontok[1]

                    # YOLO számolás
                    x_kozep = (x1 + x2) / 2 / kep_szelesseg
                    y_kozep = (y1 + y2) / 2 / kep_magassag
                    szelesseg = abs(x2 - x1) / kep_szelesseg
                    magassag = abs(y2 - y1) / kep_magassag

                    ki.write(f"{osztaly_id} {x_kozep:.6f} {y_kozep:.6f} {szelesseg:.6f} {magassag:.6f}\n")


print("Átalakítás indul...")
atalakit_yolo(train_fajlok, 'train')
atalakit_yolo(val_fajlok, 'val')

# --- data.yaml LÉTREHOZÁSA ---
yaml_szoveg = f"""
train: train/images
val: val/images

nc: {len(osztalyok)}
names: {osztalyok}
"""

yaml_utvonal = os.path.join(kimeneti_mappa, 'data.yaml')

with open(yaml_utvonal, 'w') as f:
    f.write(yaml_szoveg.strip())

print(f"\nKÉSZ! Az adatok itt vannak: {kimeneti_mappa}")
print("A data.yaml most már relatív útvonalakat használ!")