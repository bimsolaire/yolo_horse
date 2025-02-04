import cv2
import torch
from ultralytics import YOLO
from collections import deque

model = YOLO('yolov8n.pt')
# Utilise un Hardware dédié à l'IA si disponible
if torch.cuda.is_available():
    model.to('cuda')
else:
    model.to('cpu')

# Ouvrir la vidéo d'entrée
cap = cv2.VideoCapture('cheval.mp4')
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la vidéo 'cheval.mp4'.")
    exit()

# Récupérer les propriétés de la vidéo (taille, FPS)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Créer l'encodeur vidéo et le flux de sortie
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('cheval_update.mp4', fourcc, fps, (width, height))

# Durée (en secondes) après laquelle on coupe s'il n'y a pas de cheval
cut_duration_seconds = 2.0
# Calcul du nombre de frames correspondant à la durée choisie
MAX_NO_HORSE_FRAMES = int(fps * cut_duration_seconds)

# Buffer pour stocker temporairement les frames sans cheval
buffer_frames = deque()
# Compteur de frames consécutives sans cheval
no_horse_count = 0

# Compteur de frames, on travaillera que sur les frames pairs
frame_index = 0
# On mémorise la dernière valeur de horse_detected pour les frames impairs
last_detection = False

# Définir une nouvelle résolution pour optimiser l'analyse
new_width, new_height = 640, 360

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # On redimensionne la frame pour une analyse plus rapide
    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Détection YOLO sur les frames pairs redimensioné
    if frame_index % 2 == 0:
        results = model.predict(resized_frame, conf=0.5, verbose=False)
        horse_detected = False
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 17:
                    horse_detected = True
                    break
            if horse_detected:
                break
        last_detection = horse_detected
    else:
        # Sur les frames impairs : Pas de détection -> on réutilise la dernière valeur connue
        horse_detected = last_detection

    # Si on détecte un cheval, on vide le buffer dans la vidéo
    if horse_detected:
        while buffer_frames:
            out.write(buffer_frames.popleft())
        # On écrit aussi la frame courante
        out.write(frame)
        # On réinitialise le compteur
        no_horse_count = 0
    else:
        # On ajoute +1 au compteur
        no_horse_count += 1
        # On aoute la frame sans cheval dans le buffer
        buffer_frames.append(frame)

        # Si on dépasse 2 secondes sans cheval, on coupe (on vide le buffer)
        if no_horse_count > MAX_NO_HORSE_FRAMES:
            buffer_frames.clear()
            # On continue simplement sans écrire ces frames, mais on peut créer un nouveau flux vidéo où on vide le
            # buffer pour en faire un extrait avant de le supprimer

    # On ajoute +1 au compteur de frame
    frame_index += 1

cap.release()
out.release()

print("La vidéo 'cheval_update.mp4' a été générée")
