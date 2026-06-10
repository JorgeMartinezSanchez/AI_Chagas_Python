from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_dir, classes, transform, usar_mascara=False):
        self.transform     = transform
        self.usar_mascara  = usar_mascara
        self.samples       = []

        for label_idx, clase in enumerate(classes):
            carpeta = os.path.join(data_dir, clase)
            for root, dirs, files in os.walk(carpeta):
                for fname in files:
                    ruta = os.path.join(root, fname)
                    self.samples.append((ruta, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ruta, label = self.samples[idx]
        try:
            img = Image.open(ruta).convert("RGB")
            if self.usar_mascara:
                img = self.aplicar_mascara(img)
            return self.transform(img), label
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

    def aplicar_mascara(img_pil):

        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        hsv    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)


        lower    = np.array([0, 0, 0])
        upper    = np.array([180, 255, 80])
        mascara  = cv2.inRange(hsv, lower, upper)
        resultado = cv2.bitwise_and(img_cv, img_cv, mask=mascara)

        return Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))