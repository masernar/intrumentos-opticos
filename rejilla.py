#Transformada de Fresnel

# Importar librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft # Usaremos scipy.fft para fft2 y ifft2 que son eficientes y manejan el shift

#Crear campos ópticos de entrada
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

        # El campo se inicializa como cero (completamente oscuro)
        self.field = np.zeros((size, size), dtype=np.complex128)

        # Creamos las coordenadas físicas de la rejilla
        # El centro físico de la rejilla estará en (0, 0)
        grid_span = size * pixel_pitch
        coords = np.linspace(-grid_span / 2, grid_span / 2, size)
        self.x_coords, self.y_coords = np.meshgrid(coords, coords)

    def add_aperture(self, shape, center=(0, 0), size=None, value=1.0 + 0j):
        """
        Añade una apertura de una forma específica al campo.
        El valor se multiplica por la máscara de la forma, no la reemplaza.
        """
        if size is None:
            raise ValueError("El tamaño (size) debe ser especificado.")

        # --- MÁSCARAS BINARIAS (0 o 1) ---
        if shape.lower() == 'circ':
            radius = size / 2.0
            mask_shape = (self.x_coords - center[0])**2 + (self.y_coords - center[1])**2 < radius**2
            # El campo se suma para permitir superposiciones
            self.field += mask_shape.astype(np.complex128) * value

        elif shape.lower() == 'rect':
            width, height = size
            mask_shape = (np.abs(self.x_coords - center[0]) < width / 2.0) & \
                         (np.abs(self.y_coords - center[1]) < height / 2.0)
            self.field += mask_shape.astype(np.complex128) * value

        # --- MÁSCARAS GRADUALES (valores entre 0 y 1) ---
        elif shape.lower() == 'gauss':
            # Para un Gaussiano, 'size' representa el radio de la viga (beam waist, w0)
            # donde la amplitud cae a 1/e (~37%).
            w0 = size
            # Coordenadas relativas al centro
            x_rel = self.x_coords - center[0]
            y_rel = self.y_coords - center[1]
            # Perfil Gaussiano de amplitud
            mask_shape = np.exp(-(x_rel**2 + y_rel**2) / (w0**2))
            self.field += mask_shape.astype(np.complex128) * value

        elif shape.lower() == 'sinc':
            # Para un Sinc, 'size' representa el ancho del lóbulo principal.
            # Usamos np.sinc(x) que es sin(pi*x)/(pi*x)
            width = size
            # Coordenadas relativas normalizadas
            x_norm = (self.x_coords - center[0]) / width
            y_norm = (self.y_coords - center[1]) / width
            # Perfil Sinc 2D (producto de dos Sinc 1D)
            mask_shape = np.sinc(x_norm) * np.sinc(y_norm)
            self.field += mask_shape.astype(np.complex128) * value

        elif shape.lower() == 'ronchi':
            # Para una rejilla Ronchi, 'size' representa el periodo 'd' en metros.
            periodo = size
            # Generamos una onda cuadrada usando la función seno y sign.
            # np.sin(2 * np.pi * self.x_coords / periodo) crea una onda senoidal.
            # np.sign() la convierte en una onda cuadrada (-1 y 1).
            # Sumamos 1 y dividimos por 2 para que sea 0 y 1.
            mask_shape = (np.sign(np.sin(2 * np.pi * self.x_coords / periodo)) + 1) / 2
            # El campo se suma para permitir superposiciones
            self.field += mask_shape.astype(np.complex128) * value

        else:
            raise ValueError(f"Forma '{shape}' no reconocida. Use 'circ', 'rect', 'gauss', o 'sinc'.")

    def plot_intensity(self, title="Intensidad del Campo"):
        """Visualiza la intensidad (amplitud al cuadrado) del campo."""
        plt.figure(figsize=(8, 8))
        # Usamos np.abs(self.field)**2 para la intensidad
        plt.imshow(np.abs(self.field)**2, cmap='gray',
                   extent=[self.x_coords.min(), self.x_coords.max(),
                           self.y_coords.min(), self.y_coords.max()])
        plt.title(title)
        plt.xlabel("Posición X (m)")
        plt.ylabel("Posición Y (m)")
        plt.colorbar(label="Intensidad (unidades arbitrarias)")
        plt.show()

    def plot_phase(self, title="Fase del Campo"):
        """Visualiza la fase del campo."""
        plt.figure(figsize=(8, 8))
        # Usamos np.angle para obtener la fase
        plt.imshow(np.angle(self.field), cmap='twilight_shifted',
                   extent=[self.x_coords.min(), self.x_coords.max(),
                           self.y_coords.min(), self.y_coords.max()])


        plt.title(title)
        plt.xlabel("Posición X (m)")
        plt.ylabel("Posición Y (m)")
        plt.colorbar(label="Fase (radianes)")
        plt.show()

    def add_image(self, filepath, target_width, center=(0, 0), value=1.0 + 0j):
        """
        Carga una imagen desde un archivo y la añade al campo como una máscara de amplitud.

        Args:
            filepath (str): Ruta al archivo de la imagen (PNG, JPG, etc.).
            target_width (float): Ancho físico deseado para la imagen en la rejilla (en metros).
                                  La altura se escalará para mantener la proporción.
            center (tuple): Coordenadas (x, y) donde se centrará la imagen (en metros).
            value (complex): Valor complejo que modulará la imagen. Por defecto es 1.0 (amplitud pura).
        """
        try:
            # 1. Cargar la imagen y convertirla a escala de grises (modo 'L')
            img = Image.open(filepath).convert('L')
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta: {filepath}")
            return

        # 2. Convertir la imagen a un array de NumPy y normalizarla (0-255 -> 0.0-1.0)
        img_array = np.array(img) / 255.0

        # 3. Calcular las dimensiones de la imagen en píxeles de nuestra rejilla
        original_width_px, original_height_px = img.size
        aspect_ratio = original_height_px / original_width_px

        target_width_px = int(target_width / self.pixel_pitch)
        target_height_px = int(target_width_px * aspect_ratio)

        # 4. Redimensionar la imagen a los píxeles calculados usando un filtro de alta calidad
        # Creamos una nueva imagen de Pillow desde nuestro array normalizado para redimensionar
        img_to_resize = Image.fromarray((img_array * 255).astype(np.uint8))
        resized_img = img_to_resize.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)

        # Convertimos la imagen redimensionada de vuelta a un array normalizado
        resized_array = np.array(resized_img) / 255.0

        # 5. Calcular la posición para pegar la imagen en la rejilla principal
        # Convertimos el centro en metros a un offset en píxeles desde el centro de la rejilla
        center_x_px_offset = int(center[0] / self.pixel_pitch)
        center_y_px_offset = int(center[1] / self.pixel_pitch)

        # El centro de la rejilla está en (size/2, size/2)
        paste_center_x = self.size // 2 + center_x_px_offset
        paste_center_y = self.size // 2 + center_y_px_offset

        # Coordenadas de la esquina superior izquierda donde empezamos a pegar
        start_x = paste_center_x - target_width_px // 2
        start_y = paste_center_y - target_height_px // 2

        # Coordenadas de la esquina inferior derecha
        end_x = start_x + target_width_px
        end_y = start_y + target_height_px

        # Seguridad: Asegurarse de que la imagen no se sale de la rejilla
        if start_x < 0 or end_x > self.size or start_y < 0 or end_y > self.size:
            print("Advertencia: La imagen es demasiado grande o está descentrada y excede los límites de la rejilla. Será recortada.")
            # Esta parte se podría hacer más robusta con clipping, pero por ahora lo dejamos así.

        # 6. Pegar la imagen en el campo óptico
        self.field[start_y:end_y, start_x:end_x] += resized_array.astype(np.complex128) * value

def propagate_fresnel_fft(input_field_obj, z):
    """
    Propaga un campo óptico usando la Transformada de Fresnel con una sola FFT (Corregido).

    Args:
        input_field_obj (OpticalField): El objeto OpticalField de entrada.
        z (float): La distancia de propagación en metros.

    Returns:
        OpticalField: Un nuevo objeto OpticalField con el campo propagado.
    """
    # 1. Recuperar parámetros
    U0 = input_field_obj.field
    size = input_field_obj.size
    dx = input_field_obj.pixel_pitch
    wavelength = input_field_obj.wavelength
    k = 2 * np.pi / wavelength

    # 2. Crear las rejillas de coordenadas espaciales (ya están en el objeto)
    x = input_field_obj.x_coords
    y = input_field_obj.y_coords

    # 3. Multiplicar el campo de entrada por la fase parabólica de entrada
    phase_in = np.exp(1j * k / (2 * z) * (x**2 + y**2))
    U_in_phased = U0 * phase_in

    # 4. Calcular la FFT
    # ANTES de la FFT, movemos el origen del centro a la esquina
    U_in_phased_shifted = fft.ifftshift(U_in_phased)
    A = fft.fft2(U_in_phased_shifted)
    # DESPUÉS de la FFT, movemos el origen de la esquina de vuelta al centro
    A_shifted = fft.fftshift(A)

    # 5. Multiplicar por los factores de propagación y fase de salida

    # Factor de propagación global
    global_factor = np.exp(1j * k * z) / (1j * wavelength * z)

    # Fase parabólica de salida (ya está centrada, igual que x e y)
    phase_out = np.exp(1j * k / (2 * z) * (x**2 + y**2))

    # Factor de escala debido a la discretización de la integral de Fourier
    # Este factor es crucial y depende de cómo se definen las coordenadas
    # de la FFT. Para la relación que buscamos, es (dx^2).
    scaling_factor = dx**2

    # El campo final se obtiene multiplicando todos los términos
    U_out = global_factor * phase_out * scaling_factor * A_shifted

    # 6. Crear el objeto de salida
    output_field_obj = OpticalField(size, dx, wavelength)
    output_field_obj.field = U_out

    return output_field_obj

# --- FIN DE LA IMPLEMENTACIÓN DE FRESNEL FFT ---


if __name__ == "__main__":
    # --- PARÁMETROS FÍSICOS DEL PROBLEMA ---
    WAVELENGTH = 633e-9   # 633 nm
    PERIODO_REJILLA = 100e-6 # 10 pares/mm -> d = 100 µm

    # --- PARÁMETROS DE LA SIMULACIÓN ---
    # Necesitamos píxeles pequeños para resolver bien el periodo de 100 µm.
    # Usemos al menos 10 píxeles por periodo.
    PIXEL_PITCH = 2e-6   # 2 µm por píxel (50 píxeles por periodo, ¡excelente!)
    GRID_SIZE = 2048     # Una rejilla grande para ver varios periodos

    # --- CÁLCULO ANALÍTICO DE LA DISTANCIA DE TALBOT ---
    z_talbot = 2 * PERIODO_REJILLA**2 / WAVELENGTH
    print(f"Predicción Analítica:")
    print(f"Distancia de Talbot (z_T): {z_talbot * 100:.2f} cm")
    print(f"Media distancia de Talbot (z_T/2): {z_talbot/2 * 100:.2f} cm")
    print(f"Cuarto de distancia de Talbot (z_T/4): {z_talbot/4 * 100:.2f} cm")
    print("-" * 30)

    # --- 1. Crear el campo de entrada: La Rejilla Ronchi ---
    campo_entrada = OpticalField(GRID_SIZE, PIXEL_PITCH, WAVELENGTH)
    campo_entrada.add_aperture('ronchi', size=PERIODO_REJILLA)

    print("Visualizando la Rejilla Ronchi de entrada...")
    campo_entrada.plot_intensity("Rejilla Ronchi de Entrada (d=100 µm)")

    # --- 2. Propagar y estudiar la formación de autoimágenes ---
    # Vamos a probar las distancias que calculamos analíticamente.

    # Propagación a z_T / 4
    print(f"\nPropagando a z_T/4 = {z_talbot/4 * 100:.2f} cm...")
    campo_zt_4 = propagate_fresnel_fft(campo_entrada, z_talbot / 4)
    campo_zt_4.plot_intensity(f"Intensidad a z_T/4 ({z_talbot/4 * 100:.2f} cm)")

    # Propagación a z_T / 2
    print(f"\nPropagando a z_T/2 = {z_talbot/2 * 100:.2f} cm...")
    campo_zt_2 = propagate_fresnel_fft(campo_entrada, z_talbot / 2)
    campo_zt_2.plot_intensity(f"Intensidad a z_T/2 ({z_talbot/2 * 100:.2f} cm)")

    # Propagación a z_T (Distancia de Talbot completa)
    print(f"\nPropagando a z_T = {z_talbot * 100:.2f} cm...")
    campo_zt_completa = propagate_fresnel_fft(campo_entrada, z_talbot)
    campo_zt_completa.plot_intensity(f"Intensidad a z_T ({z_talbot * 100:.2f} cm) - Autoimagen")

     # Propagación a 2*z_T (Distancia de Talbot completa)
    print(f"\nPropagando a z_T = {2*z_talbot * 100:.2f} cm...")
    campo_zt_completa = propagate_fresnel_fft(campo_entrada, z_talbot)
    campo_zt_completa.plot_intensity(f"Intensidad a z_T ({2*z_talbot * 100:.2f} cm) - Autoimagen")