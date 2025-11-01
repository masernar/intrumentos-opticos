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
        (Basado en tu código original)
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
            label = "Log(Intensidad)"
        else:
            data_to_plot = intensity
            label = "Intensidad"

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

def propagate_chirped_FT(input_field_obj, f, delta):
    """
    Aplica el modelo de la "casi-FT" (Ec. 8) para la Rama Inferior.
    Devuelve un NUEVO objeto OpticalField.
    """
    # 1. Parámetros
    d_o = f + delta
    
    # 2. Fase Interna (chirp)
    fase_interna_data = np.exp(1j * input_field_obj.k / (2 * d_o) * \
                             (input_field_obj.x_coords**2 + input_field_obj.y_coords**2))
    
    # 3. Campo de entrada "chirpeado"
    U_in_chirped = input_field_obj.field * fase_interna_data
    
    # 4. Aplicar la FFT (centrada)
    U_out_data = fft.fftshift(fft.fft2(fft.ifftshift(U_in_chirped)))
    
    # 5. Fase Externa (chirp)
    fase_externa_data = np.exp(-1j * input_field_obj.k * delta / (2 * f**2) * \
                             (input_field_obj.x_coords**2 + input_field_obj.y_coords**2))

    # 6. Crear el nuevo objeto de campo de salida
    output_field_obj = OpticalField(input_field_obj.size,
                                    input_field_obj.pixel_pitch,
                                    input_field_obj.wavelength)
    
    # 7. Asignar el campo final (con fase externa)
    output_field_obj.field = U_out_data * fase_externa_data
    
    return output_field_obj