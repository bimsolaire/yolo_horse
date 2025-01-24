import cv2
from ultralytics import YOLO

# Charger un modèle YOLOv8 pré-entraîné (COCO)
model = YOLO('yolov8n.pt')

# Ouvrir la vidéo d'entrée
cap = cv2.VideoCapture('cheval.mp4')
if not cap.isOpened():
    print("Erreur: Impossible d'ouvrir la vidéo 'cheval.mp4'.")
    exit()

# Récupérer les propriétés de la vidéo (taille, FPS) pour créer la sortie
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Créer l'encodeur vidéo (codec) et le flux de sortie
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('cheval_update.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        # Fin du fichier vidéo
        break

    # Effectuer la détection sur la frame
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Vérifier si un cheval (class=17 en COCO) est détecté sur la frame
    horse_detected = False
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 17:  # 17 correspond à 'horse' dans le dataset COCO
                horse_detected = True
                break
        if horse_detected:
            break

    # Si un cheval est détecté dans la frame, on l'écrit dans la nouvelle vidéo
    if horse_detected:
        out.write(frame)

# Libérer les ressources
cap.release()
out.release()

print("La vidéo 'cheval_update.mp4' a été générée avec les frames contenant un cheval.")