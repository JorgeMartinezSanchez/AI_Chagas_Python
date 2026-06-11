import customtkinter as ctk
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageTk
import threading
import os
from model import ModelLoader, CLASSES, DEVICE


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ChagasDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Chagas - Tiempo Real")
        self.root.geometry("1400x800")
        
        self.camera_running = False
        self.cap = None
        self.current_frame = None
        self.prediction_lock = threading.Lock()
 
        self.loader = None
        self.load_model()

        self.create_widgets()

        self.update_camera()
        
    def load_model(self):
        """Carga el modelo usando ModelLoader"""
        try:
            self.loader = ModelLoader("mejor_modelo.pth")
            self.loader.load()
            self.model_loaded = True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            self.model_loaded = False

    def auto_predict(self):
        """Realiza predicción automática"""
        if self.current_frame is not None and self.model_loaded:
            if self.prediction_lock.acquire(blocking=False):
                try:
                    resultado, confianza, probs = self.predecir(self.current_frame)
                    if resultado:
                        self.root.after(0, self.update_results, resultado, confianza, probs)
                finally:
                    self.prediction_lock.release()
    
    def predecir(self, image):
        """Predice usando el loader"""
        if not self.model_loaded or self.loader is None:
            return None, None, None
        
        try:
            return self.loader.predict(image)
        except Exception as e:
            print(f"Error en predicción: {e}")
            return None, None, None
    
    def create_widgets(self):
        """Crea todos los widgets de la interfaz"""
        
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
 
        left_title = ctk.CTkLabel(left_frame, text="Vista de Cámara", font=("Arial", 20, "bold"))
        left_title.pack(pady=10)
        

        self.video_label = ctk.CTkLabel(left_frame, text="", width=640, height=480)
        self.video_label.pack(pady=10, padx=10)

        camera_controls = ctk.CTkFrame(left_frame)
        camera_controls.pack(pady=10)
        
        self.start_btn = ctk.CTkButton(camera_controls, text="Iniciar Cámara", 
                                       command=self.start_camera, 
                                       fg_color="green", hover_color="darkgreen",
                                       width=150)
        self.start_btn.pack(side="left", padx=5)
        
        self.stop_btn = ctk.CTkButton(camera_controls, text="Detener Cámara", 
                                      command=self.stop_camera,
                                      fg_color="red", hover_color="darkred",
                                      width=150, state="disabled")
        self.stop_btn.pack(side="left", padx=5)
        
        self.capture_btn = ctk.CTkButton(camera_controls, text="Capturar y Predecir", 
                                         command=self.capture_and_predict,
                                         fg_color="blue", hover_color="darkblue",
                                         width=150)
        self.capture_btn.pack(side="left", padx=5)

        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=(10, 0))

        right_title = ctk.CTkLabel(right_frame, text="Resultados", font=("Arial", 20, "bold"))
        right_title.pack(pady=10)

        result_frame = ctk.CTkFrame(right_frame, corner_radius=15, fg_color="#1a1a1a")
        result_frame.pack(fill="x", padx=20, pady=20)
        
        self.result_label = ctk.CTkLabel(result_frame, text="Esperando...", 
                                         font=("Arial", 32, "bold"), 
                                         text_color="white")
        self.result_label.pack(pady=20)
        
        self.confidence_label = ctk.CTkLabel(result_frame, text="", 
                                            font=("Arial", 18), 
                                            text_color="gray")
        self.confidence_label.pack(pady=(0, 20))
        
        probs_frame = ctk.CTkFrame(right_frame, corner_radius=15)
        probs_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        probs_title = ctk.CTkLabel(probs_frame, text="Probabilidad por Clase", 
                                   font=("Arial", 16, "bold"))
        probs_title.pack(pady=10)

        self.progress_chagas = ctk.CTkProgressBar(probs_frame, width=300, height=20)
        self.progress_chagas.pack(pady=5)
        self.progress_chagas.set(0)
        
        self.label_chagas = ctk.CTkLabel(probs_frame, text="chagas: 0%", 
                                         font=("Arial", 14))
        self.label_chagas.pack()
        
        self.progress_no_chagas = ctk.CTkProgressBar(probs_frame, width=300, height=20)
        self.progress_no_chagas.pack(pady=5)
        self.progress_no_chagas.set(0)
        
        self.label_no_chagas = ctk.CTkLabel(probs_frame, text="no_chagas: 0%", 
                                           font=("Arial", 14))
        self.label_no_chagas.pack()

        info_frame = ctk.CTkFrame(right_frame, corner_radius=15)
        info_frame.pack(fill="x", padx=20, pady=10)
        
        info_title = ctk.CTkLabel(info_frame, text="Información", 
                                  font=("Arial", 16, "bold"))
        info_title.pack(pady=10)
        
        info_text = f"""
• Modelo: {self.loader.architecture if self.loader and self.model_loaded else 'No cargado'}
• Clases: chagas / no_chagas
• Tamaño entrada: 256x256
• Dispositivo: {DEVICE}
• Predicción: Tiempo real
        """
        
        info_label = ctk.CTkLabel(info_frame, text=info_text, 
                                  font=("Arial", 12), 
                                  justify="left")
        info_label.pack(pady=10)
        
        # Indicador de modelo
        if self.model_loaded:
            model_status = ctk.CTkLabel(info_frame, text="Modelo cargado correctamente", 
                                       font=("Arial", 12), text_color="green")
        else:
            model_status = ctk.CTkLabel(info_frame, text="Error al cargar el modelo", 
                                       font=("Arial", 12), text_color="red")
        model_status.pack(pady=5)
    
    def start_camera(self):
        """Inicia la cámara"""
        if not self.camera_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: No se puede acceder a la cámara")
                return
            
            self.camera_running = True
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()
    
    def capture_frames(self):
        """Captura frames de la cámara en un thread separado"""
        while self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
            else:
                break
    
    def capture_and_predict(self):
        """Captura manual"""
        if self.current_frame is not None and self.model_loaded:
            resultado, confianza, probs = self.predecir(self.current_frame)
            if resultado:
                self.update_results(resultado, confianza, probs)

    def capture_loop(self):
        """Loop de captura"""
        while self.camera_running:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.auto_predict()
            else:
                break
    
    def update_results(self, resultado, confianza, probs):
        """Actualiza UI"""
        self.result_label.configure(text=resultado.upper())
        self.result_label.configure(text_color="red" if resultado == "chagas" else "green")
        self.confidence_label.configure(text=f"Confianza: {confianza:.1f}%")
        
        if probs:
            self.progress_chagas.set(probs['chagas'] / 100)
            self.label_chagas.configure(text=f"chagas: {probs['chagas']:.1f}%")
            self.progress_no_chagas.set(probs['no_chagas'] / 100)
            self.label_no_chagas.configure(text=f"no_chagas: {probs['no_chagas']:.1f}%")
    
    def update_camera(self):
        """Actualiza el frame de video"""
        if self.camera_running and self.current_frame is not None:
            frame_display = self.current_frame.copy()
            frame_display = cv2.resize(frame_display, (640, 480))
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=imgtk)
            self.video_label.image = imgtk
        
        self.root.after(30, self.update_camera)
    
    def stop_camera(self):
        """Detiene cámara"""
        self.camera_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.video_label.configure(image="")
        self.video_label.image = None
    
    def on_closing(self):
        """Cierre"""
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = ctk.CTk()
    app = ChagasDetectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()