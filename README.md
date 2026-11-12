# <ins>Măsurarea Automată a Parametrilor Biometrici Fetali</ins>

## Descrierea proiectului

&emsp;&emsp;Scopul acestui proiect este crearea unei aplicații care să automatizeze procesul de măsurare a parametrilor biometrici ai fătului pe baza imaginilor cu ecografie utilizând algoritmi de învățare profundă și de procesare a imaginilor.

&emsp;&emsp;Parametrii biometrici care urmează să fie măsurați sunt:
- HC - Circumferința capului fătului
- BPD - Diametrul biparietal
- AC - Circumferința abdominală a fătului
- FL - Lungimea femurului

&emsp;&emsp;Rolul proiectului este de a automatiza procedura de măsurare și obținerea unor rezultate care pot să fie reproduse.

## Analiza literaturii de specialitate

&emsp;&emsp;În vederea documentării și analizării tematicii proiectului am selectat cinci articole care cuprind informații relevante legate de aplicarea metodelor de învățare profundă în analiza imaginilor cu ecografii ale fătului.

| Nr | Autori(i)/An | Titlul articolului | Aplicație/<br>Domeniu | Tehnologii<br>utilizate | Metodologie | Rezultate | Limitări | Comentarii suplimentare |
|---|---|---|---|---|---|---|---|---|
| 1 | Szymon P lotka<br>Adam Klasa<br>Aneta Lisowska<br>Joanna Seliga-Siwecka<br>Micha l Lipa<br>Tomasz Trzci´nski<br>Arkadiusz Sitek<br>2022 0 | Deep Learning Fetal Ultrasound Video Model Match Human Observers in Biometric Measurements | Imagistică, medicală, ecografie obstetricală. | CNN multitask de tip U-Net, cu ConvLSTM în bottleneck pentru informația temporală, attention gates, deep supervision și hărți de trăsături multi-scală. | 700 de videoclipuri de la sarcini cuprinse între 15-38 de săptămâni, cu split 80% antrenare- 20% trestare.<br>S-au ales cadre cu scor de clasificare >0.9 pentru plan standard, iar cel mai bun cadru s-a ales printr-o medie ponderată (scor, măsurătoare, congruența elipsei) | Comparativ cu alte CNN-uri: IoU 0.905, Dice 0.962, acuratețe 0.975 și F1 0.96.<br>Eroare de măsurare: HC 2.9mm, AC 3.8 mm, FL 0.8mm (sub pragul clinic de approx 15%)<br>Concordanță (ICC): 0.96-0.98 între observatori.<br>Estimări: eroare de 0.05 săptămâni pentru vârstă și 25g pentru greutate.<br>Viteză: approx o secundă | Date de la un singur producător de aparate (GE) și o singură instituție, deci generalizarea este incertă. Fiecare imagine de antrenare a fost adnotată de un singur specialist, iar adevărul de referință este subiectiv.Apar erori în cazuri cu umbre sau mișcare. GA șigreutatea nu se calculează dacă lipsește unul dintre cele 3 planuri standard. | Este prima lucrare care compară un model deep learning cu sonografiști pe videoclipuri ecografice complete, nu pe cadre izolate.<br>Programele existente sunt semi-automate și cer operatorului să aleagă cadrul corect în timp ce FUVAI îl alege singur. |
| 2 | Cesar Diaz-Parga, Marta Nunez-Garcia, Maria J. Carreira, Gabriel Bernardino, and Nicolas Vila-Blanco<br>14 ianuarie 2026 | AGE-US: automated gestational age estimation based on fetal ultrasound images | Imagistică medicală și obstetrică.<br>Estimare automată a vârstei gestaționale din trimestrul III, prin măsurători biometrice, HC, AC, FL, BPD | Deep learning, U-Net, transfer learning, hărți de distanță pentru femur, algoritmul Watershed, ellipse fitting, ecuatia Hadlock cu 4 parametri | Pipeline în 3 pași care au scopul de a imita practica clinică.<br>Segmentarea capui și abdomenului, localizarea capetelor femurului prin utilizarea unei hartă de distanță, calul GA prin aplicarea ecuatiei Hadlock.<br>DataSet: IMPACT(114 studii). | GA, MAE=0.76 săptămâni și MAPE=2.3%.<br>Erori procentuale la HC,BPD și AC sunt sub variabilitatea inter-observator(4.9%-11%).<br>Encoderul comun reduce parametrii cu 25% fără diferențe semnificative. | AC are cea mai mare eroare, 1.207 cm, aspect ce este cauzat de contrastul scăzut la limitele abdomenului.<br>FL are MAPE=5.4% peste intervalul inter-observator, din cauză că structura este mică și punctele sunt pe diafază.<br>SET DE DATE MIC | Cod și greutăți publice pe GitLab CiTIUS.<br>Este mai interpretabilă decât metodele end-to-end precum DeepGA. |
| 3 | Szymon Płotka, Michal K. Grzeszczyk, Robert Brawura-Biskupski-Samaha, Paweł Gutaj , Michał Lipa, Tomasz Trzciński , Ivana Išgum , Clara I. Sánchez, Arkadiusz Sitek | BabyNet++: Fetal birth weight prediction using biometry multimodal data acquired less than 24 hours before delivery | Imagistică, medicală, ecografie obstetrică, medicină perinatală | Rețea hibridă CNN+ Transformer, bazată pe 3D ResNet-18, în care ultimele două module reziduale sunt înlocuite cu Residual Transformer Module (RTM) cu atenție multi-head 3D ăi codare pozițională temporală | 194 paciente din 4 centre, cu 582 de videoclipuri (3 per parcientă cu 134.264 de cadre) de la aparate GE. Adevărul de referință este greutatea reală la naștere.Videoclipurile sunt împărțite în segmente de 16 cadre (128x128), iar predicția finală este media predicțiilor segmentelor. | Față de clinicieni: MAE 179 \~ 19f față de 188 \~ 24g, MAPE 5.1% față de 5.4%.<br>A fost mai bun în 3 din 4 centre, iar în centrul A diferența nu a fost semnificativă (p=0.09).<br>Față de alte metodeȘ a fost cel mai bun din comparație (DAFT 201g, BabyNet 285g, 3D ResNet-18 302g, ViViT 390g).<br>Ablație: adăugarea modulului DAFT aduce cel mai mare câștig (MAEde la 302g la 198g), iar combinația RTM+DAFT dă cel mai bun rezultat. Părți fetale: videoclipurile cu abdomenul singur dau 175 \~ 17g, aproape la fel de bine ca toate cele trei părți. | Set de date mic (194 de paciente) și test pe doar 38. Un singur producător de aparate (GE), doar sarcini cu făt unic și doar paciente caucaziene. Erori mai mari la greutăți extreme (sub 2500g și peste 4000g), pentru că aceste cazuri sunt rare în date. | Este prima metodă care prezice greutatea fetală la naștere. Extinde modelul anterior BabyNet, care folosea doar imagini.<br>Nu depinde de un plan standard perfect, ceea ce contează aproape de naștere, când imaginile sunt mai greu de obținut.<br>Este gândit ca instrument de sprijin pentru clinicienii mai puțin experimentați, nu ca înlocuitor, și poate ajuta la alegerea tipului de naștere. |
| 4 | Płotka, S.S. / 2024 | Enhancing prenatal care through deep learning | Imagistică medicală prenatală: ecografie fetală și chirurgie fetoscopică. | Ecografii video(GE Healthcare),CNN,Tranformers, modul DAFT(Dynamic Affine Feature Map Transform) pentru date clinice. | Analiză video 2D+t, multi-task: detectarea planurilor standard și măsurarea HC, BPD, AC, FL; fuziune imagine + date clinice | Măsurători comparabile cu ale medicilor experți; estimarea greutății din abdomen fără diferență semnificative față de metoda manuală. | Date care provin din partea unui singur producător și doar de la o populație; utilizează doar sarcini speciale, sănătoase și se observă lipsa de explicabilitate a unor modele. | Relevant pentru proiect fiind util pentru automatizrea analizelor medicale.<br>Ar putea să fie îmbunătățit prin extragerea datelor din mai multe centre și dispozitive, hărți de atenție(Grad-CAM) care au rolul de verificare a măsurătorilor. |
| 5 | Hyunwoo Cho , Dongju Kim , Sunyeob Chang , Jinbum Kang , Yangmo Yoo | A system-on-chip solution for deep learning-based automatic fetal biometric measurement | Medicină prenatală - măsurători biometrice automate din ecografii fetale. | Rețea de segmentare H-BiSeNet, Deep Learning, CNN, TensorFlow, Vitis AI, System-on-Chip(SoC) Xilinx Zynq UltraScale+(DPU + APU) | Segmentare capului, abdomenului și femurului cu o singură rețea, apoi măsurate prin elipsa(HC, AC) și puncte cheie (BPD,FL) folosind H-BiSetNet. | Segmentare mai precisă decât cea făcută folosind arhitectura U-Net și DeepLabv3+, cu un model mult mai mic;<br>Corelație ridicată cu măsurătorile manuale r:= 0.983ș procesare de 4,25 FPS pe SoC care este de aproximativ 10 ori mai rapidă decât doar pe procesor. | Resurse hardware limitate, viteza este mai mică decât cea de achizițe a ecografului.<br>Formula aproximativă pentru circumferință poate să introducă erori.<br>Nu se utilizează video ci imagini statice. | Soluție potrivită pentru ecografii portabile, însă poate fi îmbunătățită prin creșterea vitezei și testarea pe date mai diverse. |

&emsp;&emsp;Pe baza literaturii analizate mai sus propunem aplicarea algoritmului de segmentare a structurilor fătului prin utilizarea arhitecturii U-Net și apoi calcularea parametrilor biometrici prin algoritmi de procesare a imaginilor

&emsp;&emsp;Această arhitectură permite identificarea structurilor anatomice prin combinarea caracteristicilor generale ale imaginii cu detaliile spațiale de care este nevoie pentru delimitarea contururilor.

&emsp;&emsp;Alegerea U-Net este motivată de simplitatea relativă a implementării, disponibilitatea soluțiilor deja existente și open-source, dar și posibilitatea adaptării modelului la imaginile ecografice fetale.

## Componente principale:

&emsp;&emsp;Soluția propusă este organizată modular, fiecare componentă având un rol specific în procesarea imaginilor și calcularea parametrilor necesari proiectului.

![Schema bloc](schema_bloc.jpeg)

1. Date de intrare:<br>
   Imagini ecografice 2D ale capului, abdomenului și femurului fetal achiziționate în timpul examinării prenatale.
2. Preprocesare:<br>
   Pregătește imaginile prin redimensionare, normalizarea intensității pixelilor și reducerea zgomotului pentru a avea date uniforme.
3. Arhitectura U-Net:<br>
   Rețea neuronală convoluțională pentru segmentare, este formată dintr-un encoder și un decoder care sunt legate între ele prin skip-connections. Encoder-ul extrage caracteristicile extrase la nivelul cel mai profund al rețelei. Decoder-ul reconstruiește informația spațială prin operații de upsampling și convoluție. Skip-connections-urile transferă informațiile între encoder și decoder pentru păstrarea detaliilor necesare segmentării. Bottleneck-ul
4. Postprocesare:<br>
   Extrage contururile structurilor, le aproximează prin elipse și detectează punctele cheie precum: capetele femurului sau extremitățile diametrului biparietal.
5. Modul calcul biometric:<br>
   Calculează parametrii HC, BPD, AC și FL pe baza elipselor și a punctelor cheie folosindu-se de factorul de calibrare mm/pixel cu scopul de a transforma distanțele din pixeli în milimetri.
6. Output:<br>
   Imaginea segmentată cu structurile evidențiate și valorile numerice ale parametrilor biometrici utile pentru estimarea vârstei gestaționale și a greutății fetale.

## Tehnologii propuse:

&emsp;&emsp;Pentru dezvoltarea aplicației ne propunem să folosim următoarele:

&emsp;&emsp;1. Python - limbajul principal de programare;<br>
&emsp;&emsp;2. TensorFlow / Keras - are rolul de a implementa și antrena modelul U-Net;<br>
&emsp;&emsp;3. OpenCV - pentru preprocesare și postprocesare a imaginilor;<br>
&emsp;&emsp;4. NumPy - pentru calculul numeric;<br>
&emsp;&emsp;5. Streamlit - pentru dezvoltarea unei interfețe grafice ușor de folosit care va avea rolul de a încărca imaginile și de a obține rezultatele.<br>

## Etapa intermediară (50%)

&emsp;&emsp;Obiectivul acestei etape a fost măsurarea automată a circumferinței craniene (HC) pe setul de date HC18: imagine ecografică → segmentarea capului cu U-Net → mască binară → contur și elipsă → HC în milimetri. BPD, AC, FL și interfața grafică rămân pentru etapa următoare.

&emsp;&emsp;Cele două părți au fost dezvoltate și testate independent, apoi integrate. Interfața dintre ele este: **mască binară 0/1 + pixel size + dimensiunea originală a imaginii**.

![Arhitectura etapei intermediare](figuri/arhitectura.png)

### Contribuții

**Neamțu Malina Maria (segmentare U-Net)**

| Task | Ce s-a realizat | Fișiere |
|---|---|---|
| P1.1 | eliminarea dependențelor de `__main__`; modulele se pot importa fără efecte secundare | `unet.py`, `evaluate.py`, `testUnet.py` |
| P1.2 | configurație reproductibilă: seed fix, verificarea perechilor imagine–mască, hash al perechilor | `config.py` |
| P1.3 | împărțire train / validation / test după pacient (565 / 161 / 80 de pacienți, adică 701 / 198 / 100 de imagini), verificată și salvată | `config.py`, `split_info_3.txt` |
| P1.4 | metricile Dice și IoU, testate pe cazuri simple (suprapunere completă, nulă, parțială, mască goală) | `metrice.py`, `testMetrice.py` |
| P1.5 | antrenare cu monitorizare pe validare, oprire automată după `val_dice`, istoricul antrenării salvat | `main.py`, `evaluate.py` |
| P1.6 | salvarea celui mai bun model și reîncărcarea lui fără reantrenare | `main.py`, `testLoadModel.py` |
| P1.7 | predicție independentă imagine → mască binară 0/1, readusă la rezoluția originală | `predictIndependent.py` |

**Pintilie Ioana Alexandra (calcul biometric)**

| Task | Ce s-a realizat | Fișiere |
|---|---|---|
| P2.1 | legarea fiecărei imagini de calibrarea ei (pixel size) și de HC-ul de referință din CSV; statistici despre set | `calibrare.py` |
| P2.2 | măști de referință pline 0/1 din adnotările HC18, verificate automat și vizual | `data_loader.py` (`fill_annotation`), `test_masks.py` |
| P2.3 | modul de calcul independent de model: primește mască + calibrare și întoarce HC sau motivul pentru care nu se poate calcula | `biometrie.py` (`calculeaza_hc`) |
| P2.4 | extragerea conturului și selectarea capului (aria maximă, prag de 1% din imagine), testată cu zgomot și mască goală | `biometrie.py`, `test_selectie.py` |
| P2.5 | potrivirea elipsei și verificarea vizuală pe 10 imagini | `biometrie.py`, `test_elipsa.py` |
| P2.6 | perimetrul elipsei (Ramanujan) și conversia în mm cu factori separați pe x și y, testate pe forme cunoscute și pe măști redimensionate | `biometrie.py`, `test_perimetru.py`, `test_hc.py`, `test_redimensionare.py` |
| P2.7 | evaluarea HC față de referință pe toate imaginile HC18 (MAE, eroare cu semn, eroare procentuală) | `evaluare_hc.py` |

**Comun:** integrarea celor două module (`integrare_hc.py`) și demonstrația fluxului complet (`demo.py`).

### Structura fișierelor

```
PROIECT_COD/
├── Data/HC18/                     setul de date (nu este inclus în repository)
│   ├── training_set/
│   └── training_set_pixel_size_and_HC.csv
├── data_loader.py                 citirea imaginilor și umplerea adnotărilor
├── preprocessing.py               redimensionarea la 96×128
├── config.py                      configurație, seed, împărțirea datelor
├── unet.py                        arhitectura U-Net
├── metrice.py                     Dice, IoU
├── evaluate.py                    grafice, istoricul antrenării, afișarea predicțiilor
├── main.py                        antrenarea și evaluarea modelului
├── predictIndependent.py          predicție pe o singură imagine
├── calibrare.py                   pixel size și HC de referință din CSV
├── biometrie.py                   contur, elipsă, HC în mm
├── evaluare_hc.py                 evaluarea HC pe măștile de referință
├── integrare_hc.py                HC din măștile prezise, pe setul de test
├── demo.py                        demonstrația fluxului complet
├── test*.py                       testele fiecărui modul
├── figuri/                        figurile generate
└── rezultate/                     tabelele cu rezultate
```

### Cum se rulează

1. Se instalează dependențele:
   ```
   pip install tensorflow opencv-python numpy pandas matplotlib imageio termcolor
   ```
2. Se descarcă setul HC18 și se pune în `Data/HC18/` ca în structura de mai sus.
3. Modelul antrenat nu este inclus în repository din cauza dimensiunii: **[link Google Drive, de completat]**. Se pune în folderul proiectului cu numele `unet_fetal_head.keras`. Alternativ, se antrenează cu `python main.py`.
4. Demonstrația fluxului complet: `python demo.py`
5. Evaluările:
   - segmentare: rezultatele se scriu în `validation_metrics.txt` și `test_metrics.txt` la rularea `main.py`;
   - HC pe măștile de referință: `python evaluare_hc.py`;
   - HC din măștile prezise: `python integrare_hc.py`.

### Rezultate

**Calculul HC pe măștile de referință (toate imaginile HC18)**

| Metrică | Valoare |
|---|---|
| imagini cu HC valid | 995 / 999 |
| MAE | 0.72 mm |
| eroare medie cu semn | +0.72 mm |
| eroare procentuală medie | 0.43% |
| eroare maximă | 2.01 mm |

- Pe forme geometrice cunoscute, elipsa + Ramanujan are o eroare de ~0.03%, față de ~5.7% la măsurarea directă a lungimii conturului.
- La măștile redimensionate la 96×128, conversia cu factori separați pe x și y dă erori sub 2%; un singur factor de scalare dă erori de 3–5%.
- Cele 4 imagini fără rezultat au capul parțial în afara imaginii și sunt semnalate ca invalide.

![HC calculat față de referință](figuri/p27_hc_vs_referinta.png)

**Segmentare U-Net**

&emsp;&emsp;Modelul a fost antrenat pe imagini redimensionate la 96×128, cu oprire automată după Dice-ul pe validare (răbdare 8 epoci). Antrenarea s-a oprit după 51 de epoci, iar modelul salvat este cel de la epoca 43, cu cel mai bun Dice pe validare.

| Metrică | Validare (198 imagini) | Test (100 imagini) |
|---|---|---|
| Dice | 0.972 | 0.970 |
| IoU | 0.947 | 0.944 |

&emsp;&emsp;Valorile pe validare și pe test sunt apropiate, deci modelul generalizează pe pacienți nevăzuți la antrenare.

![Evoluția antrenării](training_history.png)

**Fluxul complet (U-Net → HC) pe setul de test**

Evaluare pe primele 20 de imagini din setul de test (`integrare_test.py`):

| Metrică | Valoare |
|---|---|
| Dice mediu la rezoluția originală | 0.934 |
| MAE din masca de referință | 0.49 mm |
| MAE din masca prezisă de U-Net | 4.63 mm (5.15%) |
| imagini cu HC valid din masca prezisă | 20 / 20 |

&emsp;&emsp;Eroarea medie este dominată de două cazuri în care segmentarea a eșuat parțial: `140_HC` (Dice 0.62, HC subestimat cu 28.7 mm) și `152_HC` (Dice 0.81, HC supraestimat cu 21.0 mm). Fără aceste două imagini, MAE pe celelalte 18 este de aproximativ 2.4 mm.

Exemple din `demo.py`:

| Imagine | Dice | HC medic | HC din masca medicului | HC din masca U-Net |
|---|---|---|---|---|
| 016_HC | 0.953 | 65.0 mm | 65.4 mm | 62.4 mm |
| 031_HC | 0.901 | 67.1 mm | 67.5 mm | 61.8 mm |
| 039_HC | 0.972 | 72.8 mm | 73.1 mm | 73.2 mm |

![Elipsa din masca medicului (verde) și din masca U-Net (roșu)](figuri/integrare_exemple.png)

&emsp;&emsp;Eroarea HC din fluxul complet urmează calitatea segmentării: la un Dice de 0.97 diferența este sub 1 mm, iar la un Dice de 0.90 ajunge la ~5 mm. U-Net tinde să subestimeze marginea exterioară a craniului, ceea ce produce un HC mai mic. Pe măștile medicului, calculul geometric rămâne la sub 0.5 mm de referință, deci eroarea provine din segmentare.

### Limitări

- Segmentarea se face la rezoluție redusă (96×128), iar U-Net tinde să subestimeze marginea craniului; aceasta limitează precizia HC din fluxul complet.
- Rezultatul de 0.72 mm validează lanțul de măsurare și calibrare, nu precizia clinică: adnotările HC18 sunt elipse, iar HC-ul de referință provine din ele.
- HC este supraestimat sistematic cu ~0.7 mm, deoarece masca se umple până la marginea exterioară a liniei de adnotare.
- Capul parțial în afara imaginii nu poate fi măsurat corect.
- Toate rezultatele sunt obținute pe un singur set de date (HC18), din un singur centru, cu imagini achiziționate pe ecografe GE Voluson E8 și Voluson 730; generalizarea pe alte aparate nu a fost testată.
- Unele imagini HC18 provin din aceeași examinare; de aceea împărțirea datelor s-a făcut după pacient.

### Pașii următori

- îmbunătățirea segmentării: rezoluție mai mare, augmentarea datelor, o funcție de pierdere care include Dice;
- BPD estimat din axa mică a elipsei;
- AC cu aceeași abordare U-Net + elipsă, pe un set de date cu imagini ale abdomenului (de exemplu ACOUSLIC-AI);
- FL prin detectarea capetelor femurului;
- interfața Streamlit pentru încărcarea unei ecografii și afișarea rezultatelor.
