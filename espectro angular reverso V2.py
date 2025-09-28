# Importar libreriaas necesarias
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft # Usaremos scipy.fft para fft2 y ifft2 que son eficientes y manejan el shift


#Crear campos oppticos de entrada
class OpticalField:
    """
    Clase para crear y gestionar un campo √≥ptico complejo 2D.
    """
    def __init__(self, size, pixel_pitch, wavelength):
        """
        Inicializa la rejilla del campo √≥ptico.

        Args:
            size (int): Tama√±o de la rejilla en p√≠xeles (ej. 1024).
            pixel_pitch (float): Tama√±o del p√≠xel en metros (ej. 1e-6 para 1 ¬µm).
            wavelength (float): Longitud de onda de la luz en metros (ej. 633e-9 para HeNe).
        """
        self.size = size
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength

        # El campo se inicializa como cero (completamente oscuro)
        self.field = np.zeros((size, size), dtype=np.complex128)

        # Creamos las coordenadas f√≠sicas de la rejilla
        # El centro f√≠sico de la rejilla estar√° en (0, 0)
        grid_span = size * pixel_pitch
        coords = np.linspace(-grid_span / 2, grid_span / 2, size)
        self.x_coords, self.y_coords = np.meshgrid(coords, coords)

    def add_aperture(self, shape, center=(0, 0), size=None, value=1.0 + 0j):
        """
        A√±ade una apertura de una forma espec√≠fica al campo.
        El valor se multiplica por la m√°scara de la forma, no la reemplaza.
        """
        if size is None:
            raise ValueError("El tama√±o (size) debe ser especificado.")

        # --- M√ÅSCARAS BINARIAS (0 o 1) ---
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

        # --- M√ÅSCARAS GRADUALES (valores entre 0 y 1) ---
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
            # Para un Sinc, 'size' representa el ancho del l√≥bulo principal.
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
        plt.xlabel("Posici√≥n X (m)")
        plt.ylabel("Posici√≥n Y (m)")
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
        plt.xlabel("Posici√≥n X (m)")
        plt.ylabel("Posici√≥n Y (m)")
        plt.colorbar(label="Fase (radianes)")
        plt.show()

    def add_image(self, filepath, target_width, center=(0, 0), value=1.0 + 0j):
        """
        Carga una imagen desde un archivo y la a√±ade al campo como una m√°scara de amplitud.

        Args:
            filepath (str): Ruta al archivo de la imagen (PNG, JPG, etc.).
            target_width (float): Ancho f√≠sico deseado para la imagen en la rejilla (en metros).
                                  La altura se escalar√° para mantener la proporci√≥n.
            center (tuple): Coordenadas (x, y) donde se centrar√° la imagen (en metros).
            value (complex): Valor complejo que modular√° la imagen. Por defecto es 1.0 (amplitud pura).
        """
        try:
            # 1. Cargar la imagen y convertirla a escala de grises (modo 'L')
            img = Image.open(filepath).convert('L')
        except FileNotFoundError:
            print(f"Error: No se encontr√≥ el archivo en la ruta: {filepath}")
            return

        # 2. Convertir la imagen a un array de NumPy y normalizarla (0-255 -> 0.0-1.0)
        img_array = np.array(img) / 255.0

        # 3. Calcular las dimensiones de la imagen en p√≠xeles de nuestra rejilla
        original_width_px, original_height_px = img.size
        aspect_ratio = original_height_px / original_width_px

        target_width_px = int(target_width / self.pixel_pitch)
        target_height_px = int(target_width_px * aspect_ratio)

        # 4. Redimensionar la imagen a los p√≠xeles calculados usando un filtro de alta calidad
        # Creamos una nueva imagen de Pillow desde nuestro array normalizado para redimensionar
        img_to_resize = Image.fromarray((img_array * 255).astype(np.uint8))
        resized_img = img_to_resize.resize((target_width_px, target_height_px), Image.Resampling.LANCZOS)

        # Convertimos la imagen redimensionada de vuelta a un array normalizado
        resized_array = np.array(resized_img) / 255.0

        # 5. Calcular la posici√≥n para pegar la imagen en la rejilla principal
        # Convertimos el centro en metros a un offset en p√≠xeles desde el centro de la rejilla
        center_x_px_offset = int(center[0] / self.pixel_pitch)
        center_y_px_offset = int(center[1] / self.pixel_pitch)

        # El centro de la rejilla est√° en (size/2, size/2)
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
            print("Advertencia: La imagen es demasiado grande o est√° descentrada y excede los l√≠mites de la rejilla. Ser√° recortada.")
            # Esta parte se podr√≠a hacer m√°s robusta con clipping, pero por ahora lo dejamos as√≠.

        # 6. Pegar la imagen en el campo √≥ptico
        self.field[start_y:end_y, start_x:end_x] += resized_array.astype(np.complex128) * value

def propagate_asm(input_field, z, padding_factor=2):
    """
    Propaga un campo √≥ptico una distancia z usando el m√©todo del espectro angular.

    Args:
        input_field (OpticalField): El objeto OpticalField que contiene el campo de entrada.
        z (float): La distancia de propagaci√≥n en metros.
          padding_factor (int): Factor por el cual se aumenta el tama√±o de la rejilla
                              internamente (ej. 2 significa duplicar las dimensiones).


    Returns:
        OpticalField: Un nuevo objeto OpticalField con el campo propagado.
    """

 # --- Paso 1: Extraer parámetros y preparar el padding ---
    U_in_original = input_field.field
    lambda_ = input_field.wavelength
    dx = input_field.pixel_pitch  # El tamaño de píxel es el mismo en toda la simulación
    N_original = input_field.size
    N_padded = N_original * padding_factor

    # Crear una nueva rejilla grande e incrustar el campo original en el centro
    U_in_padded = np.zeros((N_padded, N_padded), dtype=np.complex128)
    start = (N_padded - N_original) // 2
    end = start + N_original
    U_in_padded[start:end, start:end] = U_in_original

    # --- Paso 2: Propagación en el dominio de la frecuencia (sin shifts incorrectos) ---
    # Calcular el espectro de frecuencias (f=0 está en la esquina)
    A = fft.fft2(U_in_padded)

    # Construir la Función de Transferencia (H)
    freq_coords_1d = fft.fftfreq(N_padded, dx)
    fx, fy = np.meshgrid(freq_coords_1d, freq_coords_1d)

    k = 2 * np.pi / lambda_
    
    # El término dentro de la raíz cuadrada
    # Se pone a cero si es negativo para evitar errores y manejar ondas evanescentes
    term_sqrt_sq = 1 - (lambda_ * fx)**2 - (lambda_ * fy)**2
    term_sqrt = np.sqrt(np.maximum(term_sqrt_sq, 0))

    H = np.exp(1j * k * z * term_sqrt)
    
    # Multiplicar en el dominio de la frecuencia
    A_out = A * H

    # Volver al dominio espacial
    U_out_padded = fft.ifft2(A_out)

    # --- Paso 3: Recortar la región central ---
    U_out_original = U_out_padded[start:end, start:end]

    # Devolver el resultado como un nuevo objeto OpticalField
    output_field = OpticalField(size=N_original, pixel_pitch=dx, wavelength=lambda_)
    output_field.field = U_out_original
    return output_field

#______________________________________________________________________________________________________


# --- PARÁMETROS DE LA SIMULACIÓN CORREGIDOS ---
# Usamos los parámetros de tu cámara DMM37UX226-ML
PIXEL_PITCH = 1.85e-6  # 1.85 µm
GRID_SIZE = 4096      # Potencia de 2, ligeramente mayor a la dimensión de la cámara
WAVELENGTH = 632.9e-9   #láser HeNe
D1 = 2e-3   # 2 mm
D2 = 14e-3  # 14 mm


# ==============================================================================
# PASO 1: CARGAR LAS IMÁGENES USANDO EL MÉTODO add_image
# ==============================================================================
print("Cargando imágenes de difracción en los campos ópticos...")

# Rutas a los archivos
filepath_d1 = '2mm cortado.tiff'
filepath_d2 = '14mm cortada.tiff'

# Define el ancho físico que ocupará tu holograma en la rejilla.
# Debería corresponder al tamaño del área del sensor que usaste.
# Por ejemplo, si tu recorte fue de 3500 píxeles de ancho:
ancho_fisico_holograma = 3500 * PIXEL_PITCH 

# Crear un campo óptico para el holograma en D1
campo_d1_medido = OpticalField(GRID_SIZE, PIXEL_PITCH, WAVELENGTH)
campo_d1_medido.add_image(filepath_d1, target_width=ancho_fisico_holograma)

# Crear un campo óptico para el holograma en D2
campo_d2_medido = OpticalField(GRID_SIZE, PIXEL_PITCH, WAVELENGTH)
campo_d2_medido.add_image(filepath_d2, target_width=ancho_fisico_holograma)

# Extraemos la Amplitud de los campos y la normalizamos
# np.abs() funciona sobre el campo complejo para darnos la amplitud
Amp_d1 = np.abs(campo_d1_medido.field)
Amp_d2 = np.abs(campo_d2_medido.field)

if np.max(Amp_d1) > 0 and np.max(Amp_d2) > 0:
    Amp_d1 /= np.max(Amp_d1)
    Amp_d2 /= np.max(Amp_d2)
    print("Imágenes cargadas y normalizadas correctamente.")
else:
    print("Error: Una de las imágenes parece estar completamente negra después de cargarla.")

# Opcional: Visualiza los hologramas cargados para verificar que todo está bien
campo_d1_medido.plot_intensity("Holograma Medido en D1 (2mm)")
campo_d2_medido.plot_intensity("Holograma Medido en D2 (14mm)")

# ==============================================================================
# PASO 2: FUNCIÓN PARA CREAR LA FASE ESFÉRICA
# ==============================================================================

def crear_fase_esferica(campo_optico, z_fuente):
    """
    Genera una máscara de fase correspondiente a una onda esférica.

    Args:
        campo_optico (OpticalField): El campo para el cual se genera la fase.
                                     Se usa para obtener las coordenadas y la longitud de onda.
        z_fuente (float): Distancia de la fuente puntual al plano. Es el radio de curvatura.

    Returns:
        np.ndarray: Un array 2D complejo con la máscara de fase.
    """
    k = 2 * np.pi / campo_optico.wavelength
    x = campo_optico.x_coords
    y = campo_optico.y_coords

    # Usamos la aproximación paraxial (Fórmula de Fresnel) que es muy precisa para estas distancias
    # Fase = exp(j * k * (x^2 + y^2) / (2 * z))
    termino_fase = (k * (x**2 + y**2)) / (2 * z_fuente)
    
    return np.exp(1j * termino_fase)

# ==============================================================================
# PASO 3: BÚSQUEDA Y OPTIMIZACIÓN DEL PARÁMETRO DE FASE (z_fuente)
# ==============================================================================
print("\nIniciando proceso de optimización para encontrar z_fuente...")

# Creamos un rango de distancias de fuente para probar.
# Este rango es una suposición inicial. Si el mínimo está en un extremo,
# deberás ampliar el rango. Empecemos con algo razonable.
z_fuentes_a_probar = np.linspace(1e-3, 30e-3, 100) # Probar de 1mm a 30mm en 100 pasos
errores = []

# Distancia de retro-propagación (de D2 a D1)
distancia_propagacion = D1 - D2 # Será -0.012 m

# El campo objetivo contra el que compararemos
intensidad_objetivo_d1 = Amp_d1**2

for zf in z_fuentes_a_probar:
    # 1. Crear el campo inicial en D2. Amplitud medida, fase plana.
    campo_d2 = OpticalField(GRID_SIZE, PIXEL_PITCH, WAVELENGTH)
    campo_d2.field = Amp_d2.astype(np.complex128)

    # 2. Crear y aplicar la fase esférica de prueba
    fase_prueba = crear_fase_esferica(campo_d2, zf)
    campo_d2.field *= fase_prueba
    
    # 3. Retro-propagar de D2 a D1
    campo_propagado_a_d1 = propagate_asm(campo_d2, distancia_propagacion)

    # 4. Calcular la intensidad y el error
    intensidad_calculada_d1 = np.abs(campo_propagado_a_d1.field)**2
    intensidad_calculada_d1 /= np.max(intensidad_calculada_d1) # Normalizar para comparar

    error = np.mean((intensidad_objetivo_d1 - intensidad_calculada_d1)**2)
    errores.append(error)
    print(f"Probando z_fuente = {zf*1000:.2f} mm -> Error MSE = {error:.6f}")

# Encontrar el mejor z_fuente
min_error_idx = np.argmin(errores)
z_fuente_optimo = z_fuentes_a_probar[min_error_idx]

print(f"\n¡Optimización completada!")
print(f"El valor óptimo para z_fuente es: {z_fuente_optimo * 1000:.3f} mm")

# Visualizar la curva de error
plt.figure()
plt.plot(z_fuentes_a_probar * 1000, errores)
plt.xlabel("Distancia de la fuente, z_fuente (mm)")
plt.ylabel("Error Cuadrático Medio (MSE)")
plt.title("Error de reconstrucción vs. z_fuente")
plt.grid(True)
plt.show()

# ==============================================================================
# PASO 4: RECONSTRUCCIÓN FINAL CON LA FASE ÓPTIMA
# ==============================================================================
print("\nRealizando la reconstrucción final con la fase óptima...")

# 1. Crear el campo en D1 con la amplitud medida
campo_d1_final = OpticalField(GRID_SIZE, PIXEL_PITCH, WAVELENGTH)
campo_d1_final.field = Amp_d1.astype(np.complex128)

# 2. Generar la fase esférica óptima
fase_optima = crear_fase_esferica(campo_d1_final, z_fuente_optimo)

# 3. Aplicar la corrección de fase
campo_d1_final.field *= fase_optima

# 4. Retro-propagar desde D1 hasta el plano del objeto (z=0)
distancia_reconstruccion = -D1
campo_reconstruido = propagate_asm(campo_d1_final, distancia_reconstruccion)

# 5. Visualizar la imagen reconstruida (la intensidad)
campo_reconstruido.plot_intensity(title=f"Imagen Reconstruida (z_fuente = {z_fuente_optimo*1000:.2f} mm)")

# También es útil ver la fase de la imagen reconstruida.
# Si el objeto es de solo amplitud, la fase debería ser casi plana.
campo_reconstruido.plot_phase(title=f"Fase de la Imagen Reconstruida")



