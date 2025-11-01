#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Framework para la simulación de Óptica de Fourier (Actividad 1)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft

class OpticalField:
    """
    Clase para crear y gestionar un campo óptico complejo 2D.
    """
    def __init__(self, size, pixel_pitch, wavelength):
        """
        Inicializa la rejilla del campo óptico.
        """
        self.size = size
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.k = 2 * np.pi / wavelength

        # El campo se inicializa como cero (completamente oscuro)
        self.field = np.zeros((size, size), dtype=np.complex128)

        # Creamos las coordenadas físicas de la rejilla
        grid_span = size * pixel_pitch
        coords = np.linspace(-grid_span / 2, grid_span / 2, size, endpoint=False)
        self.x_coords, self.y_coords = np.meshgrid(coords, coords)

    def add_aperture(self, shape, center=(0, 0), size=None, value=1.0 + 0j):
        """
        Añade una apertura (o valor) a la rejilla.
        """
        if size is None:
            raise ValueError("El tamaño (size) debe ser especificado.")

        if shape.lower() == 'circ':
            radius = size / 2.0
            mask_shape = (self.x_coords - center[0])**2 + (self.y_coords - center[1])**2 < radius**2
            self.field += mask_shape.astype(np.complex128) * value
        
        elif shape.lower() == 'rect':
            width, height = size
            mask_shape = (np.abs(self.x_coords - center[0]) < width / 2.0) & \
                         (np.abs(self.y_coords - center[1]) < height / 2.0)
            self.field += mask_shape.astype(np.complex128) * value
        
        elif shape.lower() == 'ronchi':
            periodo = size
            mask_shape = (np.sign(np.sin(2 * np.pi * self.x_coords / periodo)) + 1) / 2
            self.field += mask_shape.astype(np.complex128) * value
        
        # ... (puedes añadir 'gauss', 'sinc', 'image' de tu código) ...
        
        else:
            raise ValueError(f"Forma '{shape}' no reconocida.")
            
    def add_image(self, filepath, target_width, center=(0, 0), value=1.0 + 0j):
        """
        Carga una imagen desde un archivo y la añade al campo como una máscara de amplitud.
        """
        try:
            img = Image.open(filepath).convert('L') # Convertir a escala de grises
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta: {filepath}")
            return

        img_array = np.array(img) / 255.0

        original_width_px, original_height_px = img.size
        aspect_ratio = original_height_px / original_width_px

        target_width_px = int(target_width / self.pixel_pitch)
        target_height_px = int(target_width_px * aspect_ratio)
        
        if target_width_px == 0 or target_height_px == 0:
            print(f"Error: target_width ({target_width} m) es demasiado pequeño para el pixel_pitch ({self.pixel_pitch} m).")
            return

        img_to_resize = Image.fromarray((img_array * 255).astype(np.uint8))
        resized_img = img_to_resize.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)
        resized_array = np.array(resized_img) / 255.0

        center_x_px_offset = int(center[0] / self.pixel_pitch)
        center_y_px_offset = int(center[1] / self.pixel_pitch)

        paste_center_x = self.size // 2 + center_x_px_offset
        paste_center_y = self.size // 2 + center_y_px_offset

        start_x = paste_center_x - target_width_px // 2
        start_y = paste_center_y - target_height_px // 2
        end_x = start_x + target_width_px
        end_y = start_y + target_height_px

        # --- Recorte de seguridad (Clipping) ---
        # Si la imagen se sale, la recortamos para que quepa
        
        # 1. Calcular cuánto se sale
        clip_start_x = max(0, -start_x)
        clip_start_y = max(0, -start_y)
        clip_end_x = max(0, end_x - self.size)
        clip_end_y = max(0, end_y - self.size)
        
        # 2. Ajustar los arrays
        # (Si no hay recorte, clip_start_x=0 y clip_end_x=0, así que no se hace nada)
        img_slice = resized_array[clip_start_y : target_height_px - clip_end_y,
                                  clip_start_x : target_width_px - clip_end_x]
        
        field_start_x = max(0, start_x)
        field_start_y = max(0, start_y)
        field_end_x = min(self.size, end_x)
        field_end_y = min(self.size, end_y)

        # 3. Pegar la imagen (asegurándonos de que las formas coincidan)
        if img_slice.shape == self.field[field_start_y:field_end_y, field_start_x:field_end_x].shape:
             self.field[field_start_y:field_end_y, field_start_x:field_end_x] += \
                img_slice.astype(np.complex128) * value
        else:
            print("Advertencia: No se pudo pegar la imagen por un desajuste de formas (probablemente 1 píxel).")

    def apply_transmittance(self, mask_object):
        """
        Aplica una máscara de transmitancia (otro OpticalField) al campo actual.
        """
        # Multiplicación elemento a elemento
        self.field = self.field * mask_object.field

    def get_intensity(self):
        """
        Devuelve la intensidad (módulo al cuadrado) del campo.
        """
        return np.abs(self.field)**2

    def plot_intensity(self, title="Intensidad", log_scale=False):
        """Visualiza la intensidad (amplitud al cuadrado) del campo."""
        plt.figure(figsize=(8, 8))
        
        intensity = self.get_intensity()
        
        # Opción de escala logarítmica (¡muy útil para planos de Fourier!)
        if log_scale:
            # Añadimos un pequeño epsilon para evitar log(0)
            data_to_plot = np.log10(intensity + 1e-10)
            label = "Log(Intensidad) (unidades arbitrarias)"
        else:
            data_to_plot = intensity
            label = "Intensidad (unidades arbitrarias)"

        plt.imshow(data_to_plot, cmap='gray',
                   extent=[self.x_coords.min(), self.x_coords.max(),
                           self.y_coords.min(), self.y_coords.max()])
        plt.title(title)
        plt.xlabel("Posición X (m)")
        plt.ylabel("Posición Y (m)")
        plt.colorbar(label=label)
        plt.show()

# --- FIN DE LA CLASE ---

# --- FUNCIONES DE PROPAGACIÓN ---

def propagate_FT(input_field_obj):
    """
    Aplica una Transformada de Fourier (viaje de S a M1, o de M1 a O).
    Devuelve un NUEVO objeto OpticalField.
    """
    # 1. Realiza la FFT (centrada)
    # ifftshift: mueve el (0,0) del centro a la esquina
    # fft2: calcula la transformada
    # fftshift: mueve el (0,0) de la esquina de vuelta al centro
    U_out_data = fft.fftshift(fft.fft2(fft.ifftshift(input_field_obj.field)))
    
    # 2. Crea el nuevo objeto de campo de salida
    output_field_obj = OpticalField(input_field_obj.size,
                                    input_field_obj.pixel_pitch,
                                    input_field_obj.wavelength)
    
    # 3. Asigna el campo calculado
    output_field_obj.field = U_out_data
    
    return output_field_obj

# --- PROGRAMAS PRINCIPALES ---
if __name__ == "__main__":
    
    # --- Parámetros Globales de Simulación ---
    WAVELENGTH = 633e-9   # 633 nm
    PIXEL_PITCH = 5e-6    # 5 µm
    GRID_SIZE = 1024      # 1024x1024 píxeles
    FOCAL_LENGTH_f = 100e-3 # 100 mm (para L1 y L2)
    
    
    # --- 1. Definir el Objeto de Entrada (S) ---
    print("Creando objeto de entrada...")
    campo_S = OpticalField(GRID_SIZE, PIXEL_PITCH, WAVELENGTH)

    try:
        ruta_imagen = '/home/mateusi/Desktop/Transm_E01.png'
        campo_S.add_image(ruta_imagen, target_width=2e-3)
    except FileNotFoundError:
        print("ADVERTENCIA: No se encontró la imagen de prueba. Usando un círculo simple.")
        campo_S.add_aperture('circ', size=1e-3, value=1.0)
        
    campo_S.plot_intensity(title="Objeto de Entrada (S)")


    
    # === PROGRAMA 2: Simulación Rama Inferior (S -> U) ===
    def run_simulation_branch_2(input_field, f):
        print("\n--- Ejecutando Rama inferior ---")
        
        # 1. Viaje S -> U (Plano Fourier)
        print("Calculando S -> U")
        campo_U = propagate_FT(input_field)
        
        # 2. Obtener la intensidad para graficar
        intensity = campo_U.get_intensity()
        
        # --- 3. CALCULAR LA ESCALA FÍSICA CORRECTA ---
        
        # Parámetros del plano de entrada (S)
        dx = input_field.pixel_pitch  # ej. 5e-6 m
        N = input_field.size          # ej. 1024
        
        # Ancho total de la rejilla de entrada
        L_in = N * dx               
        
        # El "pixel pitch" en el espacio de FRECUENCIA (fx) es:
        dfx = 1 / L_in
        
        # El ancho total del espacio de FRECUENCIA (fx) es:
        L_fx = N * dfx # = 1 / dx
        
        # Creamos el vector de coordenadas de frecuencia (fx)
        f_coords = np.linspace(-L_fx/2, L_fx/2, N, endpoint=False)
        
        # Convertimos las coordenadas de frecuencia (fx) a
        # coordenadas FÍSICAS en la cámara (xf)
        # usando la fórmula: xf = lambda * f * fx
        xf_coords = f_coords * input_field.wavelength * f
        
        # El 'extent' para el plot es [min(x), max(x), min(y), max(y)]
        plot_extent = [xf_coords.min(), xf_coords.max(), 
                       xf_coords.min(), xf_coords.max()]
        
        # --- 4. GRAFICAR MANUALMENTE CON LA ESCALA CORRECTA ---
        plt.figure(figsize=(8, 8))
        
        data_to_plot = np.log10(intensity + 1e-10)
        label = "Log(Intensidad) (unidades arbitrarias)"

        plt.imshow(data_to_plot, cmap='gray', extent=plot_extent)
        
        plt.title("Imagen final en Cam2 (U) - Escala Física Correcta")
        plt.xlabel("Posición X en el plano de Fourier (m)")
        plt.ylabel("Posición Y en el plano de Fourier (m)")
        plt.colorbar(label=label)
        plt.show()

 
    # Ejecutar la simulación de la Rama 2
    # Pasamos FOCAL_LENGTH_f como argumento
    run_simulation_branch_2(campo_S, FOCAL_LENGTH_f)

