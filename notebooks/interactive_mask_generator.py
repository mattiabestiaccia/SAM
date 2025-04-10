import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
import torch
import os
from matplotlib.widgets import Button

class InteractiveMaskGenerator:
    def __init__(self, image_path):
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.points = []
        self.labels = []
        self.fig, self.ax = plt.subplots(figsize=(10,10))
        self.current_mode = 1  # 1 per inclusione, 0 per esclusione
        
    def on_click(self, event):
        if event.inaxes == self.ax:
            self.points.append([event.xdata, event.ydata])
            self.labels.append(self.current_mode)
            
            # Visualizza il punto
            color = 'green' if self.current_mode == 1 else 'red'
            self.ax.scatter(event.xdata, event.ydata, color=color, marker='*', s=100)
            self.fig.canvas.draw()
            
    def toggle_mode(self, event):
        self.current_mode = 1 - self.current_mode
        mode_text = "Inclusione" if self.current_mode == 1 else "Esclusione"
        print(f"Modalità: {mode_text}")
        
    def generate_mask(self):
        return np.array(self.points), np.array(self.labels)

def process_images(input_folder, output_folder, sam_checkpoint):
    # Inizializza SAM
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)
    
    # Crea la cartella di output se non esiste
    os.makedirs(output_folder, exist_ok=True)
    
    # Itera sulle immagini
    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, img_name)
            
            # Inizializza il generatore interattivo
            generator = InteractiveMaskGenerator(img_path)
            
            # Configura l'interfaccia interattiva
            generator.ax.imshow(generator.image)
            generator.fig.canvas.mpl_connect('button_press_event', generator.on_click)
            
            # Aggiungi pulsante per cambiare modalità
            ax_button = plt.axes([0.8, 0.025, 0.1, 0.04])
            btn = Button(ax_button, 'Cambia Modalità')
            btn.on_clicked(generator.toggle_mode)
            
            plt.show(block=True)
            
            # Genera la maschera
            points, labels = generator.generate_mask()
            
            if len(points) > 0:
                # Predici la maschera usando SAM
                predictor.set_image(generator.image)
                masks, _, _ = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False
                )
                
                # Salva la maschera
                mask_name = f"mask_{img_name}"
                mask_path = os.path.join(output_folder, mask_name)
                cv2.imwrite(mask_path, (masks[0] * 255).astype(np.uint8))
                
                print(f"Maschera salvata: {mask_path}")
            
            plt.close()

# Uso:
input_folder = "/home/brus/Projects/SAM/segment-anything-main/images"
output_folder = "/home/brus/Projects/SAM/segment-anything-main/images/masks"
sam_checkpoint = "/home/brus/Projects/SAM/segment-anything-main/checkpoints/sam_vit_h_4b8939.pth"

process_images(input_folder, output_folder, sam_checkpoint)