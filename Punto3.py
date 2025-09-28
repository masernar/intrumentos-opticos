#Transformada de Fresnel

# Importar librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft # Usaremos scipy.fft para fft2 y ifft2 que son eficientes y manejan el shift

#Crear campos opticos de entrada
class OpticalField:
    """
    Clase para crear y gestionar un campo optico complejo 2D.
    """
    def __init__(self, size, pixel_pitch, wavelength):
        """
        Inicializa la rejilla del campo optico.

        Args:
            size (int): Tamano de la rejilla en pixeles.
            pixel_pitch (float): Tamano del pixel en metros.
            wavelength (float): Longitud de onda de la luz en metros.
        """
        self.size = size
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength

        # El campo se inicializa como cero (completamente oscuro)
        self.field = np.zeros((size, size), dtype=np.complex128)

        # Creamos las coordenadas fisicas de la rejilla
        # El centro fisico de la rejilla estara en (0, 0)
        grid_span = size * pixel_pitch
        coords = np.linspace(-grid_span / 2, grid_span / 2, size)
        self.x_coords, self.y_coords = np.meshgrid(coords, coords)

    def add_aperture(self, shape, center=(0, 0), size=None, value=1.0 + 0j):
        """
        Anade una apertura de una forma especifica al campo.
        El valor se multiplica por la mascara de la forma, no la reemplaza.
        """
        if size is None:
            raise ValueError("El tamano (size) debe ser especificado.")

        # --- M�SCARAS BINARIAS (0 o 1) ---
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

        # --- MASCARAS GRADUALES (valores entre 0 y 1) ---
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
            # Para un Sinc, 'size' representa el ancho del lobulo principal.
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
            # Generamos una onda cuadrada usando la funcion seno y sign.
            # np.sin(2 * np.pi * self.x_coords / periodo) crea una onda senoidal.
            # np.sign() la convierte en una onda cuadrada (-1 y 1).
            # Sumamos 1 y dividimos por 2 para que sea 0 y 1.
            mask_shape = (np.sign(np.sin(2 * np.pi * self.x_coords / periodo)) + 1) / 2
            # El campo se suma para permitir superposiciones
            self.field += mask_shape.astype(np.complex128) * value

        else:
            raise ValueError(f"Forma '{shape}' no reconocida. Use 'circ', 'rect', 'gauss', o 'sinc'.")

    def plot_intensity(self, title="Intensidad del Campo"):
        plt.figure(figsize=(8, 8))
        # Usamos np.abs(self.field)**2 para la intensidad
        plt.imshow(np.abs(self.field)**2, cmap='gray',
                   extent=[self.x_coords.min(), self.x_coords.max(),
                           self.y_coords.min(), self.y_coords.max()])
        plt.title(title)
        plt.xlabel("Posicion X (m)")
        plt.ylabel("Posicion Y (m)")
        plt.colorbar(label="Intensidad (unidades arbitrarias)")
        plt.show()

    def plot_phase(self, title="Fase del Campo"):
        plt.figure(figsize=(8, 8))
        # Usamos np.angle para obtener la fase
        plt.imshow(np.angle(self.field), cmap='twilight_shifted',
                   extent=[self.x_coords.min(), self.x_coords.max(),
                           self.y_coords.min(), self.y_coords.max()])


        plt.title(title)
        plt.xlabel("Posicion X (m)")
        plt.ylabel("Posicion Y (m)")
        plt.colorbar(label="Fase (radianes)")
        plt.show()

    def add_image(self, filepath, target_width, center=(0, 0), value=1.0 + 0j):
        """
        Carga una imagen desde un archivo y la anade al campo como una mascara de amplitud.

        Args:
            filepath (str): Ruta al archivo de la imagen (PNG, JPG, etc.).
            target_width (float): Ancho fisico deseado para la imagen en la rejilla (en metros).
                                  La altura se escalara para mantener la proporcion.
            center (tuple): Coordenadas (x, y) donde se centrara la imagen (en metros).
            value (complex): Valor complejo que modulara la imagen. Por defecto es 1.0 (amplitud pura).
        """
        try:
            # 1. Cargar la imagen y convertirla a escala de grises (modo 'L')
            img = Image.open(filepath).convert('L')
        except FileNotFoundError:
            print(f"Error: No se encontro el archivo en la ruta: {filepath}")
            return

        # 2. Convertir la imagen a un array de NumPy y normalizarla (0-255 -> 0.0-1.0)
        img_array = np.array(img) / 255.0

        # 3. Calcular las dimensiones de la imagen en p�xeles de nuestra rejilla
        original_width_px, original_height_px = img.size
        aspect_ratio = original_height_px / original_width_px

        target_width_px = int(target_width / self.pixel_pitch)
        target_height_px = int(target_width_px * aspect_ratio)

        # 4. Redimensionar la imagen a los p�xeles calculados usando un filtro de alta calidad
        # Creamos una nueva imagen de Pillow desde nuestro array normalizado para redimensionar
        img_to_resize = Image.fromarray((img_array * 255).astype(np.uint8))
        resized_img = img_to_resize.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)

        # Convertimos la imagen redimensionada de vuelta a un array normalizado
        resized_array = np.array(resized_img) / 255.0

        # 5. Calcular la posici�n para pegar la imagen en la rejilla principal
        # Convertimos el centro en metros a un offset en p�xeles desde el centro de la rejilla
        center_x_px_offset = int(center[0] / self.pixel_pitch)
        center_y_px_offset = int(center[1] / self.pixel_pitch)

        # El centro de la rejilla est� en (size/2, size/2)
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
            print("Advertencia: La imagen es demasiado grande o esta descentrada y excede los limites de la rejilla. Sera recortada.")
            # Esta parte se podria hacer mas robusta con clipping, pero por ahora lo dejamos asi.

        # 6. Pegar la imagen en el campo optico
        self.field[start_y:end_y, start_x:end_x] += resized_array.astype(np.complex128) * value

def propagate_fresnel_fft(input_field_obj, z):
    """
    Propaga un campo óptico usando la Transformada de Fresnel.

    Args:
        input_field_obj (OpticalField): El objeto OpticalField de entrada.
        z (float): La distancia de propagación en metros.

    Returns:
        OpticalField: Un nuevo objeto OpticalField con el campo propagado y la escala correcta.
    """
    # 1. Recuperar parámetros de entrada
    U0 = input_field_obj.field
    size = input_field_obj.size
    dx_in = input_field_obj.pixel_pitch
    L_in = size * dx_in
    wavelength = input_field_obj.wavelength
    k = 2 * np.pi / wavelength

    # 2. Coordenadas espaciales de entrada
    x_in = input_field_obj.x_coords
    y_in = input_field_obj.y_coords

    # 3. Multiplicar por fase de entrada
    phase_in = np.exp(1j * k / (2 * z) * (x_in**2 + y_in**2))
    U_in_phased = U0 * phase_in

    # 4. Calcular la FFT centrada
    A_shifted = fft.fftshift(fft.fft2(fft.ifftshift(U_in_phased)))

    # 5. Calcular los parámetros del plano de salida PRIMERO
    dx_out = (wavelength * z) / L_in
    
    # 6. Crear el objeto de salida para tener acceso a sus coordenadas
    output_field_obj = OpticalField(size, dx_out, wavelength)
    x_out = output_field_obj.x_coords
    y_out = output_field_obj.y_coords

    # 7. Calcular los factores de propagación usando las coordenadas CORRECTAS
    global_factor = np.exp(1j * k * z) / (1j * wavelength * z)
    
    # La fase de salida AHORA usa las coordenadas del plano de salida (x_out, y_out)
    phase_out = np.exp(1j * k / (2 * z) * (x_out**2 + y_out**2))
    
    scaling_factor = dx_in**2
    
    # 8. Calcular el campo final
    U_out = global_factor * phase_out * scaling_factor * A_shifted
    
    # Asignar el campo calculado al objeto de salida
    output_field_obj.field = U_out
    
    return output_field_obj

# --- FIN DE LA IMPLEMENTACION DE FRESNEL FFT ---


if __name__ == "__main__":

    # --- PARAMETROS DE LA SIMULACION ---

    PIXEL_PITCH_1 = 5.3e-6 
    GRID_SIZE = 2048
    WAVELENGTH = 633E-9

   

    # --- 1. Crear el campo de entrada: La Rejilla Ronchi ---
    campo_entrada = OpticalField(GRID_SIZE, PIXEL_PITCH_1, WAVELENGTH)
    campo_entrada.add_image("C:/Users/SEBASTIAN/OneDrive - Universidad Nacional de Colombia/Escritorio/Nueva carpeta/Entrega_1/trans/Transm_E01.png",58e-4)

    print("Visualizando la imagen a la entrada")
    campo_entrada.plot_intensity("campo optico a la entrada")
        
    campo_salida=propagate_fresnel_fft(campo_entrada, (0.1151))
    campo_salida.plot_intensity("Campo propagado una distancia z={} m".format(0.1151))

    campo_salida=propagate_fresnel_fft(campo_entrada, (0.1446))
    campo_salida.plot_intensity("Campo propagado una distancia z={} m".format(0.1446))

    campo_salida=propagate_fresnel_fft(campo_entrada, (0.1946))
    campo_salida.plot_intensity("Campo propagado una distancia z={} m".format(0.1946))

    campo_salida=propagate_fresnel_fft(campo_entrada, (0.2446))
    campo_salida.plot_intensity("Campo propagado una distancia z={} m".format(0.2446))

    campo_salida=propagate_fresnel_fft(campo_entrada, (0.2946))
    campo_salida.plot_intensity("Campo propagado una distancia z={} m".format(0.2946))
