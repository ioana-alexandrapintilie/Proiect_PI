import cv2
from calibrare import citeste_calibrare
from biometrie import incarca, calculeaza_hc


IMAGINI = ["000_HC.png", "022_HC.png", "100_HC.png", "200_HC.png", "300_HC.png",
           "400_HC.png", "500_HC.png", "637_HC.png", "700_HC.png", "805_HC.png"]

cal = citeste_calibrare()

print(f"{'imagine':<12}{'HC ref':>10}{'original':>10}{'96x128 ok':>10}{'un factor':>11}")
for nume in IMAGINI:
    img, masca=incarca(nume)
    pixel_size,hc_ref=cal[nume]
    h_origin,w_orig=img.shape[:2]
    #micsorez masca la aceeasi dim cu iesirea din U-net
    masca_mica=cv2.resize(masca[...,0],(128,96),interpolation=cv2.INTER_NEAREST)
    #pe masca originala
    hc_orig=calculeaza_hc(masca,pixel_size,(h_origin,w_orig))["hc_mm"]
    #pe masca mica cu factori separati pe x si y
    hc_ok=calculeaza_hc(masca_mica,pixel_size,(h_origin,w_orig))["hc_mm"]
    #pe masca mica cu un singur factor de parca un pixel ar fi patrat
    factor_unic=pixel_size*w_orig/128
    hc_gresit=calculeaza_hc(masca_mica,factor_unic,(96,128))["hc_mm"]
    print(f"{nume:<12}{hc_ref:>9.2f}{hc_orig:>10.2f}{hc_ok:>11.2f}{hc_gresit:>11.2f}")