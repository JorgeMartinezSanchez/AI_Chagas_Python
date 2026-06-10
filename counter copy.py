import os
from predict import predecir

def count_images_recursive(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    total_images = 0
    processed = 0

    print(f"Buscando imágenes en: {directory}")
    print("-" * 50)

    son_chagas = 0
    son_no_chagas = 0

    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith(image_extensions):
                total_images += 1
                full_path = os.path.join(root, file_name)
                print(f"\n[{total_images}] Procesando: {full_path}")
                
                try:
                    res = predecir(full_path)
                    if res == "chagas":
                        son_chagas += 1
                    else:
                        son_no_chagas += 1
                    processed += 1
                except Exception as e:
                    print(f"Error al procesar {full_path}: {str(e)}")
                
                print("-" * 50)

    

    print(f"\nResumen final:")
    print(f"Total de imágenes encontradas: {total_images}")
    print(f"Imágenes procesadas exitosamente: {processed}")
    print(f"Son no chagas: {son_no_chagas}")
    print(f"Son chagas: {son_chagas}")

if __name__ == "__main__":
    # Limpiar la ruta de entrada
    directory = input("Enter directory path: ").strip()
    # Quitar comillas si existen
    directory = directory.strip('"').strip("'")
    # Convertir backslashes a forward slashes (opcional pero ayuda)
    directory = directory.replace('\\', '/')
    
    if os.path.isdir(directory):
        count_images_recursive(directory)
    else:
        print(f"Invalid directory: {directory}")
        print("Tips:")
        print("1. Asegúrate de que la ruta existe")
        print("2. Puedes usar forward slashes (/) en lugar de backslashes (\\)")
        print("3. No uses comillas al ingresar la ruta")
        print("4. Si la ruta tiene espacios, no necesitas comillas")