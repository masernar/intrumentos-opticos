# generar_holograma.py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft
import time

# =====================================================================================
# COPIAMOS LA MISMA CLASE Y FUNCIÓN DE PROPAGACIÓN PARA CONSISTENCIA
# =====================================================================================
class OpticalField:
    def __init__(self, size, pixel_pitch, wavelength):
        self.size = size
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.field = np.zeros((size, size), dtype=np.complex128)
        grid_span = size * pixel_pitch
        coords = np.linspace(-grid_span / 2, grid_span / 2, size)
        self.x_coords, self.y_coords = np.meshgrid(coords, coords)

def propagate_asm(input_field, z, padding_factor=1):
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
# NUEVA FUNCIÓN AUXILIAR PARA CARGAR LA IMAGEN DEL OBJETO
# =====================================================================================
def create_field_from_object_image(filepath, grid_size, pixel_pitch, wavelength):
    try:
        # Cargar la imagen y convertirla a escala de grises
        img_raw = Image.open(filepath).convert('L')
        # Normalizar los valores entre 0 y 1
        img_array = np.array(img_raw) / 255.0
    except FileNotFoundError:
        print(f"🚨 ERROR: No se encontró el archivo del objeto en '{filepath}'.")
        return None

    h, w = img_array.shape
    print(f"Imagen de objeto cargada: {w}x{h} píxeles.")

    if h > grid_size or w > grid_size:
        print(f"🚨 ERROR: La imagen del objeto ({w}x{h}) es más grande que el GRID_SIZE ({grid_size}).")
        return None

    # Centrar la imagen en una rejilla más grande (padding)
    padded_array = np.zeros((grid_size, grid_size), dtype=np.float64)
    start_h, start_w = grid_size // 2 - h // 2, grid_size // 2 - w // 2
    end_h, end_w = start_h + h, start_w + w
    padded_array[start_h:end_h, start_w:end_w] = img_array
    
    # Crear el campo óptico. La imagen es la amplitud, la fase es cero.
    field_object = OpticalField(grid_size, pixel_pitch, wavelength)
    field_object.field = padded_array.astype(np.complex128) # Asignamos la transmitancia como el campo
    
    print(f"Objeto incrustado en una rejilla de {grid_size}x{grid_size}.")
    return field_object

# =====================================================================================
# PASO PRINCIPAL: SIMULACIÓN
# =====================================================================================
if __name__ == "__main__":
    print("--- 🚀 Iniciando simulación para generar holograma ---")

    # ----------------------------------------------------------------------
    # ----- ✍️ MODIFICA ESTOS PARÁMETROS PARA LA SIMULACIÓN ✍️ -----
    # ----------------------------------------------------------------------
    # --- Parámetros Físicos (DEBEN SER IGUALES a los de tu script de reconstrucción) ---
    PIXEL_PITCH = 1.85e-6
    WAVELENGTH = 632.9e-9
    GRID_SIZE = 4096 

    # --- Parámetros de la Simulación ---
    # La imagen que quieres usar como objeto/transmitancia
    RUTA_IMAGEN_OBJETO = "/home/mateusi/Desktop/Transm_E01.png"  # <--- ¡CAMBIA ESTO! (Usa una imagen de prueba)
    
    # La distancia a la que quieres simular la captura del holograma
    Z_DISTANCIA_SIMULADA = 12e-3 # <--- ¡CAMBIA ESTO! (ej. 25 mm)

    # Dónde guardar el holograma generado
    RUTA_HOLOGRAMA_GENERADO = "/home/mateusi/Desktop/holo.png"
    # ----------------------------------------------------------------------

    # 1. Crear el campo óptico a partir de la imagen del objeto
    objeto_field = create_field_from_object_image(
        RUTA_IMAGEN_OBJETO, GRID_SIZE, PIXEL_PITCH, WAVELENGTH
    )

    if objeto_field:
        # 2. Propagar el campo hacia adelante (distancia Z positiva)
        print(f"\n--- Propagando el campo a una distancia de {Z_DISTANCIA_SIMULADA*1000:.1f} mm ---")
        start_time = time.time()
        holograma_field = propagate_asm(objeto_field, Z_DISTANCIA_SIMULADA, padding_factor=1)
        print(f"Propagación completada en {time.time() - start_time:.2f} segundos.")

        # 3. Calcular la intensidad (lo que capturaría el sensor de la cámara)
        holograma_intensidad = np.abs(holograma_field.field)**2
        
        # Normalizar la intensidad a un rango visible (0-255) para guardarla
        holograma_norm = (holograma_intensidad / np.max(holograma_intensidad)) * 255
        holograma_img = Image.fromarray(holograma_norm.astype(np.uint8))
        
        # 4. Guardar la imagen del holograma
        holograma_img.save(RUTA_HOLOGRAMA_GENERADO)
        print(f"\n✅ Holograma guardado exitosamente en: '{RUTA_HOLOGRAMA_GENERADO}'")

        # 5. Visualizar los resultados
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(np.abs(objeto_field.field)**2, cmap='gray')
        axes[0].set_title('1. Objeto Original (Transmitancia)')
        axes[0].axis('off')

        axes[1].imshow(holograma_intensidad, cmap='gray')
        axes[1].set_title(f'2. Holograma Simulado (a {Z_DISTANCIA_SIMULADA*1000:.1f} mm)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()