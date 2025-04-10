import cv2
import torch
import numpy as np
from modelo import CNN

CLASES = {0: "Circulo", 1: "Cuadrado", 2: "Triangulo"}

def preprocesar_imagen(img):
    img = cv2.resize(img, (28, 28))
    img = cv2.equalizeHist(img)
    img = (img.flatten() / 255.0 - 0.5) / 0.5
    return torch.tensor(img, dtype=torch.float32).view(1, 1, 28, 28)

def contar_figuras(model, img, frame, device):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        figura = img[y:y+h, x:x+w]
        entrada = preprocesar_imagen(figura).to(device)

        prediccion = model(entrada)
        clase = torch.argmax(prediccion).item()
        tipo_modelo = CLASES.get(clase, "Desconocido")

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, tipo_modelo, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        count += 1
    return count

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando el dispositivo: {device}")

    model = CNN().to(device)
    model.load_state_dict(torch.load("modelo_mejorado.pth", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        count = contar_figuras(model, thresh, frame, device)
        cv2.putText(frame, f"Figuras detectadas: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Detección de Figuras", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
