from PIL import Image
import os

def promedio_tamanio(directorio):
    anchos, altos = [], []
    for root, dirs, files in os.walk(directorio):
        for fname in files:
            ruta = os.path.join(root, fname)
            try:
                img = Image.open(ruta)
                ancho, alto = img.size
                anchos.append(ancho)
                altos.append(alto)
            except Exception:
                pass

    if not anchos:
        print("No se encontraron imágenes.")
        return

    print(f"Imágenes encontradas: {len(anchos)}")
    print(f"Ancho promedio:  {sum(anchos) / len(anchos):.0f} px")
    print(f"Alto promedio:   {sum(altos)  / len(altos) :.0f} px")
    print(f"Ancho mínimo/máximo: {min(anchos)} / {max(anchos)}")
    print(f"Alto mínimo/máximo:  {min(altos)}  / {max(altos)}")

promedio_tamanio(r"C:\Users\JORGE MARTINEZ S\Desktop\UCB\2026-1\Inteligencia Artificial\Segundo Bloque\detector_de_vinchucas\training_datasets")