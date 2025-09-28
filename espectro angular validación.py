# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import pearsonr # Para calcular la correlación

# =============================================================================
# TU CÓDIGO ORIGINAL (sin modificaciones)
# =============================================================================
class OpticalField:
    """
    Clase para crear y gestionar un campo óptico complejo 2D.
    """
    def __init__(self, size, pixel_pitch, wavelength):
        """
        Inicializa la rejilla del campo óptico.

        Args:
            size (int): Tamaño de la rejilla en píxeles (ej. 1024).
            pixel_pitch (float): Tamaño del píxel en metros (ej. 1e-6 para 1 µm).
            wavelength (float): Longitud de onda de la luz en metros (ej. 633e-9 para HeNe).
        """
        self.size = size
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength
        self.field = np.zeros((size, size), dtype=np.complex128)
        grid_span = size * pixel_pitch
        coords = np.linspace(-grid_span / 2, grid_span / 2 - pixel_pitch, size)
        self.x_coords, self.y_coords = np.meshgrid(coords, coords)

    def add_aperture(self, shape, center=(0, 0), size=None, value=1.0 + 0j):
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
        elif shape.lower() == 'gauss':
            w0 = size
            x_rel = self.x_coords - center[0]
            y_rel = self.y_coords - center[1]
            mask_shape = np.exp(-(x_rel**2 + y_rel**2) / (w0**2))
            self.field += mask_shape.astype(np.complex128) * value
        elif shape.lower() == 'sinc':
            width = size
            x_norm = (self.x_coords - center[0]) / width
            y_norm = (self.y_coords - center[1]) / width
            mask_shape = np.sinc(x_norm) * np.sinc(y_norm)
            self.field += mask_shape.astype(np.complex128) * value
        else:
            raise ValueError(f"Forma '{shape}' no reconocida. Use 'circ', 'rect', 'gauss', o 'sinc'.")

    def plot_intensity(self, title="Intensidad del Campo"):
        plt.figure(figsize=(8, 8))
        plt.imshow(np.abs(self.field)**2, cmap='gray',
                   extent=[self.x_coords.min(), self.x_coords.max(),
                           self.y_coords.min(), self.y_coords.max()])
        plt.title(title)
        plt.xlabel("Posición X (m)")
        plt.ylabel("Posición Y (m)")
        plt.colorbar(label="Intensidad")
        plt.show()

    def plot_phase(self, title="Fase del Campo"):
        plt.figure(figsize=(8, 8))
        plt.imshow(np.angle(self.field), cmap='twilight_shifted',
                   extent=[self.x_coords.min(), self.x_coords.max(),
                           self.y_coords.min(), self.y_coords.max()])
        plt.title(title)
        plt.xlabel("Posición X (m)")
        plt.ylabel("Posición Y (m)")
        plt.colorbar(label="Fase (radianes)")
        plt.show()

def propagate_asm(input_field, z, padding_factor=2):
    """
    Propaga un campo óptico una distancia z usando el método del espectro angular.
    """
    U_in_original = input_field.field
    lambda_ = input_field.wavelength
    dx_original = input_field.pixel_pitch
    N_original = input_field.size
    N_padded = N_original * padding_factor
    dx_padded = dx_original
    U_in_padded = np.zeros((N_padded, N_padded), dtype=np.complex128)
    start = (N_padded - N_original) // 2
    end = start + N_original
    U_in_padded[start:end, start:end] = U_in_original
    A_shifted = np.fft.fft2(U_in_padded)
    freq_coords_1d = np.fft.fftfreq(N_padded, dx_padded)
    fx, fy = np.meshgrid(freq_coords_1d, freq_coords_1d)
    k = 2 * np.pi / lambda_
    term_sqrt = 1 - (lambda_ * fx)**2 - (lambda_ * fy)**2
    mask = term_sqrt >= 0
    H = np.zeros((N_padded, N_padded), dtype=np.complex128)
    H[mask] = np.exp(1j * k * z * np.sqrt(term_sqrt[mask]))
    A_out_padded = np.fft.ifft2(A_shifted * H) # Combiné dos pasos aquí para eficiencia
    U_out_original = A_out_padded[start:end, start:end]
    output_field = OpticalField(size=N_original, pixel_pitch=dx_original, wavelength=lambda_)
    output_field.field = U_out_original
    return output_field


# =============================================================================
# CÓDIGO AÑADIDO PARA LA VALIDACIÓN
# =============================================================================

def analytical_gaussian_beam(w0, wavelength, z, x_coords, y_coords):
    """
    Calcula el campo complejo de un haz gaussiano teórico propagado una distancia z.
    """
    # Parámetros del haz
    k = 2 * np.pi / wavelength
    z_R = np.pi * w0**2 / wavelength  # Rango de Rayleigh

    # Parámetros dependientes de z
    w_z = w0 * np.sqrt(1 + (z / z_R)**2)
    # Evitar división por cero en z=0
    R_z = z * (1 + (z_R / z)**2) if z != 0 else np.inf
    zeta_z = np.arctan(z / z_R) # Fase de Gouy

    # Coordenadas radiales
    r_sq = x_coords**2 + y_coords**2

    # Amplitud del campo
    amplitude = (w0 / w_z) * np.exp(-r_sq / w_z**2)

    # Fase del campo
    phase = np.exp(-1j * (k * z + k * r_sq / (2 * R_z) - zeta_z))

    return amplitude * phase

# --- 1. PARÁMETROS DE LA SIMULACIÓN ---
GRID_SIZE = 1024
PIXEL_PITCH = 5e-6 # 5 µm
WAVELENGTH = 633e-9
PROPAGATION_DISTANCE = 1e-3 # 5 cm

# Parámetros del haz gaussiano inicial
w0 = 250e-6 # 250 µm de radio de cintura

# --- 2. CREAR EL CAMPO DE ENTRADA Y PROPAGARLO NUMÉRICAMENTE ---
print("Ejecutando simulación numérica...")
campo_gauss_in = OpticalField(GRID_SIZE, PIXEL_PITCH, WAVELENGTH)
campo_gauss_in.add_aperture('gauss', size=w0)

# Propagar con tu función
campo_numerico_out = propagate_asm(campo_gauss_in, PROPAGATION_DISTANCE, padding_factor=2)
print("Simulación numérica completada.")

# --- 3. CALCULAR EL CAMPO TEÓRICO ---
print("Calculando solución analítica...")
campo_analitico_array = analytical_gaussian_beam(w0, WAVELENGTH, PROPAGATION_DISTANCE,
                                                 campo_gauss_in.x_coords, campo_gauss_in.y_coords)

# Guardar el resultado en un objeto OpticalField para usar sus métodos de ploteo
campo_analitico_out = OpticalField(GRID_SIZE, PIXEL_PITCH, WAVELENGTH)
campo_analitico_out.field = campo_analitico_array
print("Cálculo analítico completado.")

# --- 4. COMPARACIÓN VISUAL ---

# Comparación de Intensidad 2D
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im0 = axes[0].imshow(np.abs(campo_numerico_out.field)**2, cmap='inferno',
                     extent=[campo_numerico_out.x_coords.min(), campo_numerico_out.x_coords.max(),
                             campo_numerico_out.y_coords.min(), campo_numerico_out.y_coords.max()])
axes[0].set_title("Intensidad Numérica (ASM)")
fig.colorbar(im0, ax=axes[0])
plt.xlabel("Posición X (m)")
plt.ylabel("Posición Y (m)")

im1 = axes[1].imshow(np.abs(campo_analitico_out.field)**2, cmap='inferno',
                     extent=[campo_analitico_out.x_coords.min(), campo_analitico_out.x_coords.max(),
                             campo_analitico_out.y_coords.min(), campo_analitico_out.y_coords.max()])
axes[1].set_title("Intensidad Analítica (Teórica)")
fig.colorbar(im1, ax=axes[1])
plt.xlabel("Posición X (m)")
plt.ylabel("Posición Y (m)")
plt.show()

# Comparación de perfiles 1D (corte por el centro)
centro_idx = GRID_SIZE // 2
plt.figure(figsize=(12, 5))

# Perfil de intensidad
plt.subplot(1, 2, 1)
plt.plot(campo_numerico_out.x_coords[0, :], np.abs(campo_numerico_out.field[centro_idx, :])**2, 'b-', label='Numérico (ASM)')
plt.plot(campo_analitico_out.x_coords[0, :], np.abs(campo_analitico_out.field[centro_idx, :])**2, 'r--', label='Analítico', linewidth=2)
plt.title('Perfil de Intensidad (Corte Central)')
plt.xlabel('Posición X (m)')
plt.ylabel('Intensidad')
plt.legend()
plt.grid(True)



# --- 5. ANÁLISIS CUANTITATIVO: CORRELACIÓN DE PEARSON ---
# Extraemos los datos numéricos y analíticos como arrays 1D
intensidad_numerica_flat = np.abs(campo_numerico_out.field).flatten()
intensidad_analitica_flat = np.abs(campo_analitico_out.field).flatten()

# Calculamos la correlación
corr_intensidad, _ = pearsonr(intensidad_numerica_flat, intensidad_analitica_flat)

print("\n--- ANÁLISIS DE CORRELACIÓN ---")
print(f"Coeficiente de correlación de Pearson para la Intensidad: {corr_intensidad:.10f}")
