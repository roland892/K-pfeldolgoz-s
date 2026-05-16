import os
from ultralytics import YOLO

def yolo_tanitas_inditasa():
    # --- ALAP ÚTVONAL ---
    # Megnézzük, hol van ez a script(FeladatP)
    alap_utvonal = os.path.dirname(os.path.abspath(__file__))

    # --- FÁJLOK ELÉRÉSI ÚTJA ---
    modell_utvonal = os.path.join(alap_utvonal, 'yolov8s.pt')
    yaml_utvonal = os.path.join(alap_utvonal, 'YOLOData', 'data.yaml')

    # --- ELLENŐRZÉS ---
    if not os.path.exists(modell_utvonal):
        print(f"HIBA: Nem találom a modellt itt: {modell_utvonal}")
        return None

    if not os.path.exists(yaml_utvonal):
        print(f"HIBA: Nem találom a data.yaml fájlt itt: {yaml_utvonal}")
        return None

    # --- MODELL BETÖLTÉSE ---
    modell = YOLO(modell_utvonal)

    # --- TANÍTÁS INDÍTÁSA ---
    print("Tanítás elindult...")

    eredmeny = modell.train(
        data=yaml_utvonal,
        epochs=10, 
        imgsz=200,
        batch=2,
        device='cpu',  # ha van GPU: '0'
        cache=True,

        # Augmentációk (adat "variálás")
        mixup=0.2,
        mosaic=0.3,
        copy_paste=0.3,

        # Tanulási beállítások
        optimizer='AdamW',
        lr0=0.0005, #meghatározza a modell mekkora lépéseket frissiti a sulyokat a tanitás elején/ tanulási ráta
        lrf=0.01, #a tanítás végére kezdőérték mekkora töredékre csokkenjen
        warmup_epochs=5,  

        # Színek módosítása
        hsv_h=0.01,
        hsv_s=0.3,
        hsv_v=0.2,

        # Geometriai módosítások
        degrees=15.0,     # forgatás
        translate=0.1,    # eltolás
        scale=0.4,        # méretezés
        shear=2.0,        # nyírás
        perspective=0.0008, #dőlés szög

        close_mosaic=2, # az utolso 2 kornél fejezze be

        project='erme_project',
        name='proba'
    )

    # --- LEGJOBB MODELL ELÉRÉSI ÚTJA ---
    legjobb_modell = os.path.join(eredmeny.save_dir, "weights", "best.pt")

    print(f"KÉSZ! A legjobb modell itt van: {legjobb_modell}")

    return legjobb_modell


if __name__ == "__main__":
    utvonal = yolo_tanitas_inditasa()
    print(f"Visszakapott útvonal: {utvonal}")