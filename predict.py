import torch
from model import load_model
from PIL import Image
import os
from pathlib import Path

MODEL_PATH = "mejor_modelo.pth"
INPUT_DIR = "splitted-video/"
OUTPUT_FILE = "resultados_clasificacion.txt"

def es_imagen(archivo):
    """Verifica si un archivo es una imagen"""
    extensiones = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    return Path(archivo).suffix.lower() in extensiones

def procesar_directorio_recursivo(directorio, loader):
    """
    Procesa recursivamente todas las imágenes en un directorio
    """
    resultados = []
    directorio = Path(directorio)
    
    if not directorio.exists():
        print(f"❌ El directorio {directorio} no existe")
        return resultados

    imagenes = list(directorio.rglob('*'))
    imagenes = [img for img in imagenes if es_imagen(img)]
    
    print(f"📁 Encontradas {len(imagenes)} imágenes en {directorio}")
    print("=" * 60)
    
    for i, img_path in enumerate(imagenes, 1):
        print(f"\n[{i}/{len(imagenes)}] Procesando: {img_path.relative_to(directorio)}")
        
        try:
            resultado, confianza, probs = loader.predict_from_path(str(img_path))
            
            resultados.append({
                'ruta': str(img_path),
                'nombre': img_path.name,
                'resultado': resultado,
                'confianza': confianza,
                'chagas_prob': probs['chagas'],
                'no_chagas_prob': probs['no_chagas']
            })
            
            print(f"  → {resultado.upper()} (confianza: {confianza:.1f}%)")
            print(f"     chagas: {probs['chagas']:.1f}% | no_chagas: {probs['no_chagas']:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            resultados.append({
                'ruta': str(img_path),
                'nombre': img_path.name,
                'resultado': 'ERROR',
                'confianza': 0,
                'error': str(e)
            })
    
    return resultados

def guardar_resultados(resultados, archivo_salida):
    """Guarda los resultados en un archivo"""
    with open(archivo_salida, 'w', encoding='utf-8') as f:
        f.write("RESULTADOS DE CLASIFICACIÓN\n")
        f.write("=" * 60 + "\n\n")
        
        # Separar por tipo de resultado
        chagas = [r for r in resultados if r.get('resultado') == 'chagas']
        no_chagas = [r for r in resultados if r.get('resultado') == 'no_chagas']
        errores = [r for r in resultados if r.get('resultado') == 'ERROR']
        
        f.write(f"Total: {len(resultados)} imágenes\n")
        f.write(f"🪳 Chagas: {len(chagas)}\n")
        f.write(f"✅ No Chagas: {len(no_chagas)}\n")
        f.write(f"❌ Errores: {len(errores)}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("DETALLE POR IMAGEN\n")
        f.write("=" * 60 + "\n\n")
        
        for r in resultados:
            if r.get('resultado') != 'ERROR':
                f.write(f"📷 {r['nombre']}\n")
                f.write(f"   Ruta: {r['ruta']}\n")
                f.write(f"   Resultado: {r['resultado'].upper()}\n")
                f.write(f"   Confianza: {r['confianza']:.1f}%\n")
                f.write(f"   Probabilidades: chagas={r['chagas_prob']:.1f}%, no_chagas={r['no_chagas_prob']:.1f}%\n\n")
            else:
                f.write(f"❌ {r['nombre']}\n")
                f.write(f"   Ruta: {r['ruta']}\n")
                f.write(f"   Error: {r.get('error', 'Desconocido')}\n\n")
    
    print(f"\n💾 Resultados guardados en: {archivo_salida}")

def main():
    # Cargar modelo
    print("🔧 Cargando modelo...")
    try:
        loader = load_model(MODEL_PATH)
        print(f"✅ Modelo cargado: {loader.architecture}\n")
    except Exception as e:
        print(f"❌ Error: {e}")

    resultados = procesar_directorio_recursivo(INPUT_DIR, loader)
    

    if resultados:
        print("\n" + "=" * 60)
        print("RESUMEN FINAL")
        print("=" * 60)
        
        chagas = sum(1 for r in resultados if r.get('resultado') == 'chagas')
        no_chagas = sum(1 for r in resultados if r.get('resultado') == 'no_chagas')
        errores = sum(1 for r in resultados if r.get('resultado') == 'ERROR')
        
        print(f"Total: {len(resultados)} imágenes")
        print(f"🪳 Chagas detectados: {chagas}")
        print(f"✅ No Chagas: {no_chagas}")
        print(f"❌ Errores: {errores}")
        
        guardar_resultados(resultados, OUTPUT_FILE)

if __name__ == "__main__":
    main()