import math
import cv2
import numpy as np
from biometrie import extrage_contururi,selecteaza_contur,potriveste_elipsa,perimetru_elipsa

FORME=[("cerc r=100",100,100,0),
       ("elipsa 200x120",200,120,30),
       ("elipsa 150x60",150,60,75)]
print(f"{'forma':<16}{'adevarat':>10}{'elipsa':>10}{'contur':>10}")
for nume,a,b,unghi in FORME:
    masca=np.zeros((540,800),dtype=np.uint8)
    cv2.ellipse(masca,(400,270),(a,b),unghi,0,360,1,-1)
    contur=selecteaza_contur(extrage_contururi(masca),540*800)
    (_,_),(d1,d2),_=potriveste_elipsa(contur)
    adevarat=perimetru_elipsa(2*a,2*b)
    din_elipsa=perimetru_elipsa(d1,d2)
    din_contur=cv2.arcLength(contur,True)
    print(f"{nume:<16}{adevarat:>10.1f}{din_elipsa:>10.1f}{din_contur:>10.1f}")