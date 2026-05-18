import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# Hivatalos méretek (átmérő mm-ben)
erme_meretek = {
    5: 21.2,
    10: 24.8,
    20: 26.3,
    50: 27.4,
    100: 23.8,
    200: 28.3
}

class ForintFelismero:
    """ Ez az osztály végzi az érmék keresését és mérését """
    
    def __init__(self, modell_fajl):
        # Modell betöltése
        self.modell = YOLO(modell_fajl)

    def kep_atmeretezes(self, kep, max_szelesseg=900, max_magassag=650):
        # Kép méretének beállítása
        h, w = kep.shape[:2]
        skala = min(max_szelesseg / w, max_magassag / h, 1.0)
        uj_w, uj_h = int(w * skala), int(h * skala)
        return cv2.resize(kep, (uj_w, uj_h), interpolation=cv2.INTER_AREA)

    def ertek_kinyeres(self, nev):
        # Kinyerjük a számot (pl. 'coin20' -> 20)
        szamok = ''.join(filter(str.isdigit, nev))
        return int(szamok) if szamok else 0

    def feldolgozas(self, kep_bgr, kuszob):
        # Fő feldolgozó logika
        kep_kicsi = self.kep_atmeretezes(kep_bgr.copy())
        rajzolt_kep = kep_kicsi.copy()

        # 1. Keressük meg az érméket
        eredmenyek = self.modell.predict(source=kep_kicsi, conf=kuszob, verbose=False)

        talalatok = []
        for r in eredmenyek:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bizalom = float(box.conf[0])
                nev = self.modell.names[int(box.cls[0])]
                tipp_ertek = self.ertek_kinyeres(nev)
                p_atmero = ((x2 - x1) + (y2 - y1)) / 2.0
               
                talalatok.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1 + x2) // 2, "cy": (y1 + y2) // 2,
                    "r_pixel": p_atmero / 2,
                    "pixel_atmero": p_atmero,
                    "tipp": tipp_ertek,
                    "conf": bizalom
                })

        if not talalatok:
            return rajzolt_kep, [], 0

        # 2. Skálázás (pixel -> mm)
        aranyok = []
        for t in talalatok:
            if t["conf"] > 0.55 and t["tipp"] in erme_meretek:
                aranyok.append(t["pixel_atmero"] / erme_meretek[t["tipp"]])

        if not aranyok:
            for t in talalatok:
                if t["tipp"] in erme_meretek:
                    aranyok.append(t["pixel_atmero"] / erme_meretek[t["tipp"]])

        px_per_mm = float(np.median(aranyok)) if aranyok else 1.0

        # 3. Szabályok és javítás
        vegleges_lista = []
        osszeg = 0

        for t in talalatok:
            mm = t["pixel_atmero"] / px_per_mm if px_per_mm > 0 else 0
            valodi_ertek = t["tipp"]

            # Fizikai ellenőrzés
            if t["tipp"] in [5, 20]:
                if mm > 26.8: valodi_ertek = 50 
                elif abs(mm - erme_meretek[5]) < abs(mm - erme_meretek[20]): valodi_ertek = 5
                else: valodi_ertek = 20
            elif t["tipp"] in [10, 50]:
                valodi_ertek = 10 if abs(mm - erme_meretek[10]) < abs(mm - erme_meretek[50]) else 50
            elif t["tipp"] in [100, 200]:
                valodi_ertek = 100 if abs(mm - erme_meretek[100]) < abs(mm - erme_meretek[200]) else 200

            osszeg += valodi_ertek
            vegleges_lista.append({"ertek": valodi_ertek, "mm": round(mm, 1), "eredeti": t["tipp"]})

            # Rajzolás
            cx, cy, r = t["cx"], t["cy"], int(t["r_pixel"])
            szin = (80, 230, 60) if valodi_ertek == t["tipp"] else (0, 180, 255)
            cv2.circle(rajzolt_kep, (cx, cy), r, szin, 3, cv2.LINE_AA)
            cv2.putText(rajzolt_kep, f"{valodi_ertek} Ft", (cx - r + 5, cy - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, szin, 2, cv2.LINE_AA)

        return rajzolt_kep, vegleges_lista, osszeg

class ForintApp:
    """ Ez az osztály kezeli a felhasználói felületet """
    
    def __init__(self, ablak):
        self.ablak = ablak
        self.ablak.title("Érmefelismerő")
        self.ablak.geometry("1400x900")
        self.ablak.configure(bg="#121212")

        self.felismero = ForintFelismero("best.pt")
        self.eredeti_kep = None
        self.kepernyo_kep_rgb = None
        self.zoom = 1.0
        
        self.felulet_epites()

    def felulet_epites(self):
        # Oldalsáv
        oldalsav = tk.Frame(self.ablak, bg="#1c1c1c", width=320)
        oldalsav.pack(side="left", fill="y")
        oldalsav.pack_propagate(False)

        tk.Label(oldalsav, text = "Érmefelismerő", fg="#00e5ff", bg="#1c1c1c", font=("Arial", 22, "bold")).pack(pady=(40, 10))

        # Gombok
        tk.Button(oldalsav, text="KÉP BETÖLTÉSE", command=self.kep_betoltes, bg="#2a2a2a", fg="#e0e0e0", font=("Arial", 10, "bold"), relief="flat", pady=12).pack(fill="x", padx=30, pady=5)
        tk.Button(oldalsav, text="FELISMERÉS", command=self.elemzes, bg="#333333", fg="#e0e0e0", font=("Arial", 10, "bold"), relief="flat", pady=12).pack(fill="x", padx=30, pady=5)

        # Csúszka
        tk.Label(oldalsav, text="Felismerési küszöb", fg="#888888", bg="#1c1c1c", font=("Arial", 9)).pack(anchor="w", padx=35, pady=(20, 0))
        self.csuszka = tk.Scale(oldalsav, from_=0.05, to=0.9, resolution=0.05, orient="horizontal", bg="#1c1c1c", fg="#e0e0e0", highlightthickness=0)
        self.csuszka.set(0.2)
        self.csuszka.pack(fill="x", padx=30)

        # Összegzés
        summary = tk.Frame(oldalsav, bg="#2a2a2a", padx=20, pady=20)
        summary.pack(fill="x", padx=30, pady=30)
        tk.Label(summary, text="ÖSSZESEN", fg="#888888", bg="#2a2a2a", font=("Arial", 9, "bold")).pack(anchor="w")
        self.szoveg = tk.StringVar(value="0 Ft")
        tk.Label(summary, textvariable=self.szoveg, fg="#e0e0e0", bg="#2a2a2a", font=("Arial", 32, "bold")).pack(anchor="w")

        # Lista
        self.lista = tk.Listbox(oldalsav, bg="#1c1c1c", fg="#aaaaaa", font=("Consolas", 9), borderwidth=0, highlightthickness=0)
        self.lista.pack(fill="both", expand=True, padx=20, pady=10)

        # Vászon
        self.vaszon = tk.Canvas(self.ablak, bg="#121212", highlightthickness=0)
        self.vaszon.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        self.vaszon.bind("<MouseWheel>", self.nagyit)
        self.vaszon.bind("<ButtonPress-1>", lambda e: self.vaszon.scan_mark(e.x, e.y))
        self.vaszon.bind("<B1-Motion>", lambda e: self.vaszon.scan_dragto(e.x, e.y, gain=1))

    def kep_betoltes(self):
        ut = filedialog.askopenfilename()
        if not ut: return
        self.eredeti_kep = cv2.imread(ut)
        kicsi = self.felismero.kep_atmeretezes(self.eredeti_kep)
        self.kepernyo_kep_rgb = cv2.cvtColor(kicsi, cv2.COLOR_BGR2RGB)
        self.zoom = 1.0
        self.lista.delete(0, tk.END)
        self.szoveg.set("0 Ft")
        self.megjelenit()

    def elemzes(self):
        if self.eredeti_kep is None: return
        rajz, eredmenyek, ossz = self.felismero.feldolgozas(self.eredeti_kep, self.csuszka.get())
        self.kepernyo_kep_rgb = cv2.cvtColor(rajz, cv2.COLOR_BGR2RGB)
        
        self.lista.delete(0, tk.END)
        db = {}
        for x in eredmenyek:
            val = x["ertek"]
            db[val] = db.get(val, 0) + 1
            sor = f" Találat: {val} Ft ({x['mm']}mm)"
            if val != x["eredeti"]:
                sor = f" [FIX] {x['eredeti']} -> {val} Ft"
            self.lista.insert(tk.END, sor)
        
        self.lista.insert(tk.END, "-"*25)
        for k in sorted(db):
            self.lista.insert(tk.END, f" {k} Ft: {db[k]} db")
        self.szoveg.set(f"{ossz} Ft")
        self.megjelenit()

    def megjelenit(self):
        if self.kepernyo_kep_rgb is None: return
        h, w = self.kepernyo_kep_rgb.shape[:2]
        m = (int(w * self.zoom), int(h * self.zoom))
        p = Image.fromarray(self.kepernyo_kep_rgb).resize(m, Image.LANCZOS)
        tk_p = ImageTk.PhotoImage(p)
        self.vaszon.image = tk_p
        self.vaszon.delete("all")
        self.vaszon.create_image(0, 0, image=tk_p, anchor="nw")

    def nagyit(self, e):
        self.zoom *= 1.1 if e.delta > 0 else 0.9
        self.zoom = max(0.1, min(5, self.zoom))
        self.megjelenit()

if __name__ == "__main__":
    main = tk.Tk()
    app = ForintApp(main)
    main.mainloop()
