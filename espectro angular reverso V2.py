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

def propagate_asm(input_field, z, padding_factor=2):
    """
    Propaga un campo óptico una distancia z usando el método del espectro angular.

    Args:
        input_field (OpticalField): El objeto OpticalField que contiene el campo de entrada.
        z (float): La distancia de propagación en metros.
          padding_factor (int): Factor por el cual se aumenta el tamaño de la rejilla
                              internamente (ej. 2 significa duplicar las dimensiones).


    Returns:
        OpticalField: Un nuevo objeto OpticalField con el campo propagado.
    """

 # --- Paso 1: Extraer par�metros y preparar el padding ---
    U_in_original = input_field.field
    lambda_ = input_field.wavelength
    dx = input_field.pixel_pitch  # El tama�o de p�xel es el mismo en toda la simulaci�n
    N_original = input_field.size
    N_padded = N_original * padding_factor

    # Crear una nueva rejilla grande e incrustar el campo original en el centro
    U_in_padded = np.zeros((N_padded, N_padded), dtype=np.complex128)
    start = (N_padded - N_original) // 2
    end = start + N_original
    U_in_padded[start:end, start:end] = U_in_original

    # --- Paso 2: Propagaci�n en el dominio de la frecuencia (sin shifts incorrectos) ---
    # Calcular el espectro de frecuencias (f=0 est� en la esquina)
    A = fft.fft2(U_in_padded)

    # Construir la Funci�n de Transferencia (H)
    freq_coords_1d = fft.fftfreq(N_padded, dx)
    fx, fy = np.meshgrid(freq_coords_1d, freq_coords_1d)

    k = 2 * np.pi / lambda_
    
    # El t�rmino dentro de la ra�z cuadrada
    # Se pone a cero si es negativo para evitar errores y manejar ondas evanescentes
    term_sqrt_sq = 1 - (lambda_ * fx)**2 - (lambda_ * fy)**2
    term_sqrt = np.sqrt(np.maximum(term_sqrt_sq, 0))

    H = np.exp(1j * k * z * term_sqrt)
    
    # Multiplicar en el dominio de la frecuencia
    A_out = A * H

    # Volver al dominio espacial
    U_out_padded = fft.ifft2(A_out)

    # --- Paso 3: Recortar la regi�n central ---
    U_out_original = U_out_padded[start:end, start:end]

    # Devolver el resultado como un nuevo objeto OpticalField
    output_field = OpticalField(size=N_original, pixel_pitch=dx, wavelength=lambda_)
    output_field.field = U_out_original
    return output_field

#______________________________________________________________________________________________________


#Ejemplo plano óptico de entrada


# --- PARÁMETROS DE LA SIMULACIÓN ---
PIXEL_PITCH = 1.85e-6
GRID_SIZE = 4096    
WAVELENGTH = 633E-9

limite=(GRID_SIZE*(PIXEL_PITCH*PIXEL_PITCH))/WAVELENGTH

print("z <= ", limite*100, "cm")

# --- Crear el campo óptico inicial ---
campo_in = OpticalField(size=GRID_SIZE,pixel_pitch=PIXEL_PITCH,wavelength=WAVELENGTH)
campo_in.add_image("/home/mateusi/Desktop/Inst op 4/2mm cortado.tiff",3000*PIXEL_PITCH)

# Visualizar el resultado
campo_in.plot_intensity(title="Intensidad de la apertura")

A_prop=propagate_asm(campo_in,2e-3,2)

A_prop.plot_intensity(title="intesidad campo proapagado")

