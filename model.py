"""
model.py - Modelo unificado para detector de Chagas
Este archivo contiene la definición del modelo y funciones para cargarlo
"""

import torch
import torch.nn as nn
from torchvision import transforms
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["chagas", "no_chagas"]
IMG_SIZE = 256

# Transform estándar para inferencia
STANDARD_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

class ImageNN(nn.Module):
    """Arquitectura CNN personalizada (entrenada originalmente)"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ModelLoader:
    """
    Clase para cargar y gestionar el modelo entrenado
    Detecta automáticamente la arquitectura del checkpoint
    """
    
    def __init__(self, model_path="mejor_modelo.pth"):
        self.model_path = model_path
        self.model = None
        self.device = DEVICE
        self.classes = CLASSES
        self.transform = STANDARD_TRANSFORM
        self.is_loaded = False
        self.architecture = None
        
    def load(self):
        """Carga el modelo desde el archivo"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No se encontró el modelo en: {self.model_path}")
        
        # Cargar checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        primera_key = list(checkpoint.keys())[0]
        
        print(f"Cargando modelo desde: {self.model_path}")
        print(f"Primera clave del checkpoint: {primera_key}")
        
        # Detectar arquitectura
        if 'features' in primera_key:
            print("✅ Detectado: Modelo ImageNN (personalizado)")
            self.model = ImageNN()
            self.architecture = "ImageNN"
        elif 'conv1' in primera_key:
            print("✅ Detectado: Modelo ResNet18")
            from torchvision import models
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, 2)
            self.architecture = "ResNet18"
        else:
            raise ValueError(f"Arquitectura no reconocida: {primera_key}")
        
        # Cargar pesos
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True
        
        print(f"✅ Modelo cargado exitosamente en {self.device}")
        return self
    
    def predict(self, image):
        """
        Predice la clase de una imagen (formato OpenCV BGR)
        
        Args:
            image: Imagen en formato BGR (como la de OpenCV)
            
        Returns:
            tuple: (resultado, confianza, dict_probabilidades)
        """
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado. Llame a .load() primero")
        
        import cv2
        from PIL import Image
        
        # Convertir BGR a RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Transformar
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predecir
        with torch.no_grad():
            output = self.model(tensor)
            probabilidades = torch.softmax(output, dim=1)[0]
            clase_idx = torch.argmax(probabilidades).item()
        
        resultado = self.classes[clase_idx]
        confianza = probabilidades[clase_idx].item() * 100
        probs = {
            'chagas': probabilidades[0].item() * 100,
            'no_chagas': probabilidades[1].item() * 100
        }
        
        return resultado, confianza, probs
    
    def predict_from_path(self, image_path):
        """
        Predice desde una ruta de archivo
        
        Args:
            image_path: Ruta a la imagen
            
        Returns:
            tuple: (resultado, confianza, dict_probabilidades)
        """
        from PIL import Image
        
        if not self.is_loaded:
            raise RuntimeError("Modelo no cargado. Llame a .load() primero")
        
        # Cargar imagen
        pil_image = Image.open(image_path).convert("RGB")
        
        # Transformar
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predecir
        with torch.no_grad():
            output = self.model(tensor)
            probabilidades = torch.softmax(output, dim=1)[0]
            clase_idx = torch.argmax(probabilidades).item()
        
        resultado = self.classes[clase_idx]
        confianza = probabilidades[clase_idx].item() * 100
        probs = {
            'chagas': probabilidades[0].item() * 100,
            'no_chagas': probabilidades[1].item() * 100
        }
        
        return resultado, confianza, probs
    
    def get_info(self):
        """Devuelve información del modelo"""
        return {
            'architecture': self.architecture,
            'device': str(self.device),
            'classes': self.classes,
            'model_path': self.model_path,
            'is_loaded': self.is_loaded
        }


# Función de conveniencia para carga rápida
def load_model(model_path="mejor_modelo.pth"):
    """Carga rápida del modelo (función de conveniencia)"""
    loader = ModelLoader(model_path)
    return loader.load()