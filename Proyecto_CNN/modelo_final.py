# reconocimiento_rt.py
import os
import cv2
import imutils
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "face_classifier_final.pth")

POSSIBLE_DATASET_PATHS = [os.path.join(SCRIPT_DIR, "rostros_dataset")]
IMG_SIZE = 400
PADDING = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)


class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        super(FaceCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def find_dataset_and_classes():
    for p in POSSIBLE_DATASET_PATHS:
        if os.path.isdir(p):
            classes = sorted([d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))])
            if classes:
                return p, classes
    return None, None




def build_and_load_model(model_path, class_names, device):
    num_classes = len(class_names)
    net = FaceCNN(num_classes)

    net.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
 
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el archivo de pesos en: {model_path}")

    try:
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state, strict=True)
    except Exception as e:
        print("Carga estricta fallida:", e)
        print("Intentando carga no estricta (strict=False). Si el resultado es incorrecto, reentrena o guarda checkpoint correcto.")
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state, strict=False)

    net.to(device)
    net.eval()
    return net




preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  
])




def real_time_recognition():
    dataset_path, class_names = find_dataset_and_classes()
    if dataset_path is None:
        print("No se encontró la carpeta del dataset en las rutas buscadas.")
        print("Coloca tu dataset en una carpeta llamada 'rostros_dataset' o 'dataset' o actualiza POSSIBLE_DATASET_PATHS.")
        return

    print("Dataset detectado en:", dataset_path)
    print("Clases:", class_names)


    try:
        model = build_and_load_model(MODEL_PATH, class_names, device)
    except Exception as e:
        print("Error al cargar el modelo:", e)
        return

  
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Verifica dispositivo/permiso.")
        return

    face_cascade = cv2.CascadeClassifier(
    r"E:\Semestre 7\IA\Proyecto_ia_recon\cascades\haarcascade_frontalface_default.xml"
)

    print("Iniciando reconocimiento (presiona 'q' para salir).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=800)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(80, 80)
            )

            for (x, y, w, h) in faces:
               
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

               
                x1 = max(0, x - PADDING)
                y1 = max(0, y - PADDING)
                x2 = min(frame.shape[1], x + w + PADDING)
                y2 = min(frame.shape[0], y + h + PADDING)
                face_roi = frame[y1:y2, x1:x2]

                try:
                    rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    tensor = preprocess(rgb).unsqueeze(0).to(device)  

             
                    with torch.no_grad():
                        if device.type == "cuda":
                            with torch.cuda.amp.autocast():
                                out = model(tensor)
                        else:
                            out = model(tensor)

                        probs = torch.softmax(out, dim=1).cpu().numpy()[0]

                    top_idx = int(np.argmax(probs))
                    conf = float(probs[top_idx])

 
                    top3 = np.argsort(probs)[-3:][::-1]
                    print(f"\nPredicción: {class_names[top_idx]} ({conf:.4f})")
                    for i in top3:
                        print(f"  - {class_names[i]}: {probs[i]:.4f}")


                    text = f"{class_names[top_idx]}: {conf*100:.1f}%"
                    color = (0, 255, 0) if conf > 0.80 else (0, 0, 255)
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                except Exception as e:
                    print("Error en predicción:", e)

            cv2.imshow("Reconocimiento RT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_recognition()
