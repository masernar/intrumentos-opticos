# Importar librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft
from scipy.ndimage import laplace
import time

# =====================================================================================
# CLASE Y FUNCIÓN DE PROPAGACIÓN (SIN CAMBIOS)
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

    def add_image(self, filepath, target_width, center=(0, 0), value=1.0 + 0j):
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
# FUNCIÓN AUXILIAR PARA CARGAR DATOS REALES (SIN CAMBIOS)
# =====================================================================================
def create_field_from_intensity_image(filepath, grid_size, pixel_pitch, wavelength):
    try:
        img_raw = Image.open(filepath).convert('L')
        img_array = np.array(img_raw)
    except FileNotFoundError:
        print(f" ERROR: No se encontró el archivo en '{filepath}'.")
        return None, None

    h, w = img_array.shape
    print(f"Imagen original cargada: {w}x{h} píxeles.")

    if h > grid_size or w > grid_size:
        print(f" ERROR: La imagen ({w}x{h}) es más grande que el GRID_SIZE ({grid_size}).")
        return None, None

    padded_array = np.zeros((grid_size, grid_size), dtype=img_array.dtype)
    start_h, start_w = grid_size // 2 - h // 2, grid_size // 2 - w // 2
    end_h, end_w = start_h + h, start_w + w
    padded_array[start_h:end_h, start_w:end_w] = img_array

    intensity_data = padded_array
    amplitude_data = np.sqrt(intensity_data.astype(np.float64))

    # Filtramos el término DC restando el valor medio
    amplitude_data = amplitude_data - np.mean(amplitude_data)

    field_object = OpticalField(grid_size, pixel_pitch, wavelength)
    field_object.field = amplitude_data

    print(f"Imagen incrustada y filtrada en una rejilla de {grid_size}x{grid_size}.")
    return field_object, intensity_data

# =====================================================================================
# FUNCIÓN DE MÉTRICA DE ENFOQUE (SOLO VARIANZA)
# =====================================================================================
def calcular_varianza(image):
    return np.var(image)

# =====================================================================================
# PASO 1: CONFIGURACIÓN Y CARGA
# =====================================================================================
print("--- 1. Cargando imagen de difracción y configurando parámetros ---")

# ----------------------------------------------------------------------
# -----  MODIFICA ESTOS PARÁMETROS  -----
# ----------------------------------------------------------------------
RUTA_A_TU_IMAGEN = "2mm cortado.tiff" # <--- ¡CAMBIA ESTO!
PIXEL_PITCH = 1.85e-6
WAVELENGTH = 632.9e-9
GRID_SIZE = 4096
Z_APROX_LAB = 2e-3

# Parámetros para la búsqueda de foco
rango_busqueda = 1e-3 # Rango en metros (ej. 4mm) para buscar alrededor de Z_APROX_LAB
num_pasos = 10      # Número de distancias a probar
z_min = Z_APROX_LAB - rango_busqueda / 2
z_max = Z_APROX_LAB + rango_busqueda / 2
# ----------------------------------------------------------------------

# Usamos la función para crear nuestro campo inicial
patron_difraccion_real, intensity_data = create_field_from_intensity_image(
    RUTA_A_TU_IMAGEN, GRID_SIZE, PIXEL_PITCH, WAVELENGTH
)

if patron_difraccion_real is None:
    exit()

# =====================================================================================
# PASO 2: BÚSQUEDA DE FOCO USANDO VARIANZA
# =====================================================================================
z_test_range = np.linspace(z_min, z_max, num_pasos)
print(f"\n--- 2. Iniciando búsqueda entre {z_min*100:.2f} cm y {z_max*100:.2f} cm ---")

metricas_varianza = []

start_time = time.time()
for i, z_test in enumerate(z_test_range):
    campo_reconstruido = propagate_asm(patron_difraccion_real, -z_test)
    intensidad = np.abs(campo_reconstruido.field)**2

    metricas_varianza.append(calcular_varianza(intensidad))

    print(f"Paso {i+1}/{num_pasos} | z = {-z_test*100:.2f} cm", end='\r')
print(f"\nBúsqueda completada en {time.time() - start_time:.2f} segundos.")

# =====================================================================================
# PASO 3: ANÁLISIS, GRÁFICO Y RECONSTRUCCIÓN FINAL
# =====================================================================================
print("\n--- 3. Analizando resultados y reconstrucción final ---")

# Encontrar la distancia óptima para la varianza
z_optima = z_test_range[np.argmax(metricas_varianza)]
print(f" Distancia óptima encontrada (Varianza): {z_optima*100:.3f} cm")

# Graficar la curva de enfoque
plt.figure(figsize=(10, 6))
plt.plot(z_test_range * 100, metricas_varianza, 'b-', label='Varianza')
plt.axvline(x=z_optima * 100, color='r', linestyle=':', label=f'Pico en {z_optima*100:.2f} cm')
plt.title('Métrica de Enfoque (Varianza) vs. Distancia')
plt.xlabel('Distancia de Retropropagación (cm)')
plt.ylabel('Varianza de la Intensidad')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------------------------------
# --- ✅ PASO 3.5: CORRECCIÓN DE FASE APLICADA AL CAMPO FINAL ---
# -------------------------------------------------------------------------------------
# Primero, retropropagamos el campo original a la distancia óptima encontrada.
print(f"\n--- 3.5. Retropropagando campo a z = {z_optima*100:.3f} cm y aplicando corrección de fase ---")
campo_enfocado = propagate_asm(patron_difraccion_real, -z_optima, padding_factor=2)

# Ahora, creamos y aplicamos la máscara de fase a este campo ya enfocado.
k = 2 * np.pi / WAVELENGTH
r_sq = campo_enfocado.x_coords**2 + campo_enfocado.y_coords**2
R_FUENTE=5e-2
# La máscara de fase conjugada para la corrección
# OJO: El signo de R_FUENTE puede ser positivo o negativo dependiendo de la convención
# y de si la onda es convergente o divergente. Prueba con ambos si el resultado no es el esperado.
mascara_fase = np.exp(-1j * k * r_sq / (2 * R_FUENTE))
campo_final_corregido = OpticalField(GRID_SIZE, PIXEL_PITCH, WAVELENGTH)
campo_final_corregido.field = campo_enfocado.field * mascara_fase
print(f"Corrección de fase aplicada para una fuente a {R_FUENTE*100:.1f} cm.")
# -------------------------------------------------------------------------------------

# Visualización final con el campo ya corregido
print("\nMostrando resultados finales...")
fig, axes = plt.subplots(1, 2, 3, figsize=(14, 7))
axes[0].imshow(intensity_data, cmap='gray')
axes[0].set_title('1. Patrón de Difracción (Entrada)')
axes[0].axis('off')

# Mostramos la intensidad (magnitud al cuadrado) del campo final corregido
axes[1].imshow(np.abs(campo_final_corregido.field)**2, cmap='gray')
axes[1].set_title(f'2. Reconstrucción Corregida (z={z_optima*100:.2f} cm, R={R_FUENTE*100:.1f} cm)')
axes[1].axis('off')

axes[2].imshow(np.abs(campo_enfocado.field)**2, cmap='gray')
axes[2].set_title(f'2. Reconstrucción Corregida (z={z_optima*100:.2f} cm, R={R_FUENTE*100:.1f} cm)')
axes[2].axis('off')

plt.tight_layout()
plt.show()