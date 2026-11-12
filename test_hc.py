from calibrare import citeste_calibrare
from biometrie import incarca,calculeaza_hc

IMAGINI = ["000_HC.png", "022_HC.png", "100_HC.png", "200_HC.png", "300_HC.png",
           "400_HC.png", "500_HC.png", "637_HC.png", "700_HC.png", "805_HC.png"]

cal = citeste_calibrare()
print(f"{'imagine':<12}{'pixel mm':>10}{'HC ref':>10}{'HC calc':>10}{'dif mm':>9}{'dif %':>8}")
for nume in IMAGINI:
    img,masca=incarca(nume)
    pixel_size,hc_ref=cal[nume]
    rez=calculeaza_hc(masca,pixel_size,img.shape[:2])
    hc=rez["hc_mm"]
    dif=hc-hc_ref
    print(f"{nume:<12}{pixel_size:>10.4f}{hc_ref:>10.2f}{hc:>10.2f}{dif:>+9.2f}{dif / hc_ref * 100:>+7.2f}%")