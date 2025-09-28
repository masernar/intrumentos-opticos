# Importar librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft
from scipy.ndimage import laplace
import time
import sys

# =====================================================================================
# TU CÓDIGO ORIGINAL (CLASE Y FUNCIÓN DE PROPAGACIÓN)
# =====================================================================================
class OpticalField:
    # ... (Tu clase OpticalField sin cambios aquí) ...
    def __init__(self, size, pixel_pitch, wavelength):
        self.size = size
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.field = np.zeros((size, size), dtype=np.complex128)
        grid_span = size * pixel_pitch
        coords = np.linspace(-grid_span / 2, grid_span / 2, size)
        self.x_coords, self.y_coords = np.meshgrid(coords, coords)

    def add_image(self, filepath, target_width, center=(0, 0), value=1.0 + 0j):
        # Este método es ideal para añadir 'objetos' a la simulación.
        # Lo mantendremos aquí para futuros usos, pero no para cargar el holograma.
        try:
            img = Image.open(filepath).convert('L')
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta: {filepath}")
            return
        img_array = np.array(img) / 255.0
        original_width_px, original_height_px = img.size
        aspect_ratio = original_height_px / original_width_px
        target_width_px = int(target_width / self.pixel_pitch)
        target_height_px = int(target_width_px * aspect_ratio)
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
        if start_x < 0 or end_x > self.size or start_y < 0 or end_y > self.size:
            print("Advertencia: La imagen excede los límites de la rejilla y será recortada.")
        self.field[start_y:end_y, start_x:end_x] += resized_array.astype(np.complex128) * value


def propagate_asm(input_field, z, padding_factor=1):
    # ... (Tu función propagate_asm sin cambios aquí) ...
    U_in_original = input_field.field
    lambda_ = input_field.wavelength
    dx = input_field.pixel_pitch
    N_original = input_field.size
    N_padded = N_original * padding_factor
    U_in_padded = np.zeros((N_padded, N_padded), dtype=np.complex128)
    start = (N_padded - N_original) // 2
    end = start + N_original
    U_in_padded[start:end, start:end] = U_in_original
    A = fft.fft2(U_in_padded)
    freq_coords_1d = fft.fftfreq(N_padded, dx)
    fx, fy = np.meshgrid(freq_coords_1d, freq_coords_1d)
    k = 2 * np.pi / lambda_
    term_sqrt_sq = 1 - (lambda_ * fx)**2 - (lambda_ * fy)**2
    term_sqrt = np.sqrt(np.maximum(term_sqrt_sq, 0))
    H = np.exp(1j * k * z * term_sqrt)
    A_out = A * H
    U_out_padded = fft.ifft2(A_out)
    U_out_original = U_out_padded[start:end, start:end]
    output_field = OpticalField(size=N_original, pixel_pitch=dx, wavelength=lambda_)
    output_field.field = U_out_original
    return output_field

# =====================================================================================
# --- NUEVA FUNCIÓN AUXILIAR PARA CARGAR DATOS REALES ---
# =====================================================================================

def create_field_from_intensity_image(filepath, grid_size, pixel_pitch, wavelength):
    """
    Carga una imagen de intensidad, la incrusta en el centro de una rejilla
    del tamaño especificado (padding) y devuelve un objeto OpticalField.
    """
    try:
        img_raw = Image.open(filepath).convert('L')
        img_array = np.array(img_raw)
    except FileNotFoundError:
        print(f"🚨 ERROR: No se encontró el archivo en '{filepath}'.")
        return None, None

    h, w = img_array.shape
    print(f"Imagen original cargada: {w}x{h} píxeles.")

    # Crear el lienzo o 'pad' del tamaño de la rejilla final
    padded_array = np.zeros((grid_size, grid_size), dtype=img_array.dtype)

    # Calcular las coordenadas para pegar la imagen en el centro
    start_h = grid_size // 2 - h // 2
    start_w = grid_size // 2 - w // 2
    end_h = start_h + h
    end_w = start_w + w
    
    # Pegar la imagen
    padded_array[start_h:end_h, start_w:end_w] = img_array
    
    intensity_data = padded_array

    # La amplitud es la raíz cuadrada de la intensidad. La fase se asume cero.
    amplitude_data = np.sqrt(intensity_data.astype(np.float64))

    # Crear y configurar el objeto OpticalField
    field_object = OpticalField(grid_size, pixel_pitch, wavelength)
    field_object.field = amplitude_data
    
    print(f"Imagen incrustada en una rejilla de {grid_size}x{grid_size} (padding).")
    return field_object, intensity_data


# FUNCIONES DE MÉTRICA DE ENFOQUE (SIN CAMBIOS)
def calcular_varianza(image): return np.var(image)
def calcular_laplaciano_abs(image): return np.sum(np.abs(laplace(image)))

# =====================================================================================
# --- PASO 1: CONFIGURACIÓN Y CARGA USANDO LA NUEVA FUNCIÓN ---
# =====================================================================================
print("--- 1. Cargando imagen de difracción y configurando parámetros ---")

# ----------------------------------------------------------------------
# ----- ✍️ MODIFICA ESTOS PARÁMETROS ✍️ -----
# ----------------------------------------------------------------------
RUTA_A_TU_IMAGEN = "/home/mateusi/Desktop/Inst op 4/14mm cortada.tiff" # <--- ¡CAMBIA ESTO!
PIXEL_PITCH = 1.85e-6
WAVELENGTH = 632.9e-9
GRID_SIZE = 4096 # <--- AJUSTA ESTO (debe ser potencia de 2)
Z_APROX_LAB = 14e-3 # <--- ¡CAMBIA ESTO! (en metros)
# ----------------------------------------------------------------------

# Usamos la nueva función para crear nuestro campo inicial
patron_difraccion_real, intensity_data = create_field_from_intensity_image(
    RUTA_A_TU_IMAGEN, GRID_SIZE, PIXEL_PITCH, WAVELENGTH
)

# Si la carga falló, el programa se detiene.
if patron_difraccion_real is None:
    sys.exit()

# =====================================================================================
# --- PASO 2 y 3: BÚSQUEDA DE FOCO Y ANÁLISIS (SIN CAMBIOS) ---
# =====================================================================================

# El resto del código es exactamente el mismo que en la versión anterior.
# Define el rango de búsqueda
z_min, z_max = Z_APROX_LAB * 0.8, Z_APROX_LAB * 1.2
num_pasos = 100
z_test_range = np.linspace(z_min, z_max, num_pasos)

# Bucle de retropropagación
print(f"\n--- 2. Iniciando búsqueda entre {z_min*100:.1f} cm y {z_max*100:.1f} cm ---")
metricas_varianza = []
metricas_laplaciano = []
start_time = time.time()
for i, z_test in enumerate(z_test_range):
    campo_reconstruido = propagate_asm(patron_difraccion_real, -z_test)
    intensidad = np.abs(campo_reconstruido.field)**2
    metricas_varianza.append(calcular_varianza(intensidad))
    metricas_laplaciano.append(calcular_laplaciano_abs(intensidad))
    print(f"Paso {i+1}/{num_pasos} | z = {-z_test*100:.2f} cm", end='\r')
print(f"\nBúsqueda completada en {time.time() - start_time:.2f} segundos.")

# Análisis de resultados
print("\n--- 3. Analizando resultados ---")
metricas_varianza = np.array(metricas_varianza) / np.max(metricas_varianza)
z_optima = z_test_range[np.argmax(metricas_varianza)]
print(f"🎯 Distancia de enfoque óptima encontrada: {z_optima*100:.3f} cm")

# Gráfico de métricas... (código de ploteo)
plt.figure(figsize=(10, 6))
plt.plot(z_test_range * 100, metricas_varianza, label='Varianza')
plt.axvline(x=z_optima * 100, color='g', linestyle=':', label=f'Z Óptima ({z_optima*100:.2f} cm)')
plt.title('Curva de Métrica de Enfoque vs. Distancia')
plt.xlabel('Distancia de Retropropagación (cm)'), plt.ylabel('Métrica Normalizada')
plt.legend(), plt.grid(True), plt.show()

# Reconstrucción final y visualización
reconstruccion_final = propagate_asm(patron_difraccion_real, -z_optima, padding_factor=2)
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(intensity_data, cmap='gray'), axes[0].set_title('1. Tu Patrón de Difracción (Entrada)'), axes[0].axis('off')
axes[1].imshow(np.abs(reconstruccion_final.field)**2, cmap='gray'), axes[1].set_title(f'2. Reconstrucción Enfocada (a {z_optima*100:.2f} cm)'), axes[1].axis('off')
plt.tight_layout(), plt.show()