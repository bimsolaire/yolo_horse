import cv2
from ultralytics import YOLO
from collections import deque

# Charger un modèle YOLOv8 pré-entraîné (COCO)
model = YOLO('yolov8n.pt')

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

while True:
    ret, frame = cap.read()
    if not ret:
        # Fin du fichier vidéo
        break

    # Effectuer la détection sur la frame
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Vérifier si un cheval (class=17 en COCO) est détecté
    horse_detected = False
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 17:  # 17 correspond à 'horse' dans le dataset COCO
                horse_detected = True
                break
        if horse_detected:
            break

    if horse_detected:
        # Si on détecte un cheval, on vide le buffer dans la vidéo
        while buffer_frames:
            out.write(buffer_frames.popleft())
        # On écrit aussi la frame courante
        out.write(frame)
        # On réinitialise le compteur
        no_horse_count = 0
    else:
        # On ajoute +1 au compteur
        no_horse_count += 1
        # On aoute la frame sans cheval dansle buffer
        buffer_frames.append(frame)

        # Si on dépasse 2 secondes sans cheval, on coupe (on vide le buffer)
        if no_horse_count > MAX_NO_HORSE_FRAMES:
            buffer_frames.clear()
            # On continue simplement sans écrire ces frames, on peut créer un nouveau flux vidéo où on vide le buffer
            # pour en faire un extrait avant de le supprimer

# Lorsque la vidéo se termine, on n’écrit pas le buffer
# car si le cheval n’a pas réapparu, on considère que la coupure est faite.
cap.release()
out.release()

print("La vidéo 'cheval_update.mp4' a été générée")
