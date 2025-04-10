import os
import cv2
import numpy as np
import pandas as pd

def procesar_imagenes(input_dir, output_csv):
    data = []
    labels = {
        'circulo': 0,
        'cuadrado': 1,
        'triangulo': 2
    }

    for label, value in labels.items():
        path = os.path.join(input_dir, label)

        if not os.path.exists(path):
            print(f"Directorio no encontrado: {path}")
            continue

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error al cargar imagen: {img_path}")
                continue

            img = cv2.resize(img, (28, 28))  # Redimensionar a 28x28 píxeles
            img = cv2.equalizeHist(img)  # Mejorar contraste
            img_flattened = (img.flatten() / 255.0 - 0.5) / 0.5  # Normalizar a [-1, 1]
            data.append(np.append(img_flattened, value))

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False, header=False)
    print(f"Datos guardados en {output_csv}")

if __name__ == "__main__":
    procesar_imagenes("./images", "figuras.csv")
