import os
import pandas as pd
import matplotlib.pyplot as plt
from calibrare import citeste_calibrare
from biometrie import incarca,calculeaza_hc,FOLDER_FIGURI

cal=citeste_calibrare()
randuri=[]

for nume,(pixel_size,hc_ref) in cal.items():
    img,masca=incarca(nume)
    rezultat=calculeaza_hc(masca,pixel_size,img.shape[:2])
    randuri.append({"imagine":nume,"hc_ref":hc_ref,"hc_calc":rezultat["hc_mm"],"motiv":rezultat["motiv"]})
df=pd.DataFrame(randuri)

df["eroare_mm"]=df["hc_calc"]-df["hc_ref"]
df["eroare_abs_mm"]=df["eroare_mm"].abs()
df["eroare_pct"]=df["eroare_mm"]/df["hc_ref"]*100#cat de mare e greseala comparativ cu mariea reala a capului

os.makedirs("rezultate",exist_ok=True)
df.round(2).to_csv(os.path.join("rezultate","rezultate_hc.csv"),index=False)

print(df.head(10).round(2).to_string(index=False))
print("\nRanduri in tabel:",len(df))
#pastrez doar randurile unde a putut fi calc hc
valide=df.dropna(subset=["hc_calc"])
print("\nimagini evaluate:",len(df))
print("imagini fara hc_calc",len(df)-len(valide))

print(f"\nMAE: {valide['eroare_abs_mm'].mean():.2f} mm")
print(f"eroarea medie cu semn: {valide['eroare_mm'].mean():+.2f} mm")
print(f"eroare procentuala absoluta medie: {valide['eroare_pct'].abs().mean():.2f} %")
print(f"eroare maxima: {valide['eroare_abs_mm'].max():.2f} mm")

print("\nCele mai mari 5 erori:")
cele_mai_mari=valide.sort_values("eroare_abs_mm",ascending=False).head(5)
print(cele_mai_mari.round(2).to_string(index=False))
print("\nImagini fara rezultat:")
print(df[df["hc_calc"].isna()][["imagine", "hc_ref", "motiv"]].to_string(index=False))

plt.figure(figsize=(6, 6))
plt.scatter(valide["hc_ref"], valide["hc_calc"], s=8, alpha=0.6)
plt.plot([40, 360], [40, 360], "r--", label="potrivire perfecta")
plt.xlabel("HC referinta (mm)")
plt.ylabel("HC calculat (mm)")
plt.title(f"HC calculat vs. referinta ({len(valide)} imagini)\nMAE = {valide['eroare_abs_mm'].mean():.2f} mm")
plt.legend()
plt.tight_layout()
os.makedirs(FOLDER_FIGURI, exist_ok=True)
plt.savefig(os.path.join(FOLDER_FIGURI, "p27_hc_vs_referinta.png"), dpi=150)
plt.show()