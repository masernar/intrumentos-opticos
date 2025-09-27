#Transformada de Fresnel

# Importar librer�as necesarias
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.fft as fft # Usaremos scipy.fft para fft2 y ifft2 que son eficientes y manejan el shift
from scipy.integrate import quad
from scipy.special import j0 # Función de Bessel de orden cero
import time
from scipy.stats import pearsonr
#Crear campos �pticos de entrada
class OpticalField:
    """
    Clase para crear y gestionar un campo �ptico complejo 2D.
    """
    def __init__(self, size, pixel_pitch, wavelength):
        """
        Inicializa la rejilla del campo �ptico.

        Args:
            size (int): Tama�o de la rejilla en p�xeles (ej. 1024).
            pixel_pitch (float): Tama�o del p�xel en metros (ej. 1e-6 para 1 �m).
            wavelength (float): Longitud de onda de la luz en metros (ej. 633e-9 para HeNe).
        """
        self.size = size
        self.pixel_pitch = pixel_pitch
        self.wavelength = wavelength

        # El campo se inicializa como cero (completamente oscuro)
        self.field = np.zeros((size, size), dtype=np.complex128)

        # Creamos las coordenadas f�sicas de la rejilla
        # El centro f�sico de la rejilla estar� en (0, 0)
        grid_span = size * pixel_pitch
        coords = np.linspace(-grid_span / 2, grid_span / 2, size)
        self.x_coords, self.y_coords = np.meshgrid(coords, coords)

    def add_aperture(self, shape, center=(0, 0), size=None, value=1.0 + 0j):
        """
        A�ade una apertura de una forma espec�fica al campo.
        El valor se multiplica por la m�scara de la forma, no la reemplaza.
        """
        if size is None:
            raise ValueError("El tama�o (size) debe ser especificado.")

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

        # --- M�SCARAS GRADUALES (valores entre 0 y 1) ---
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
            # Para un Sinc, 'size' representa el ancho del l�bulo principal.
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
            # Generamos una onda cuadrada usando la funci�n seno y sign.
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
        Carga una imagen desde un archivo y la añade al campo como una m�scara de amplitud.

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
            print("Advertencia: La imagen es demasiado grande o está descentrada y excede los límites de la rejilla. Será recortada.")

        # 6. Pegar la imagen en el campo �ptico
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


def calcular_difraccion_analitica_circular(L_out, N_out, radius, lam, z):
    """
    Calcula el patrón de difracción de Fresnel de una apertura circular
    evaluando numéricamente la integral de difracción. (VERSIÓN CORREGIDA)

    Args:
        L_out (float): Dimensión física del lado del plano de observación [m].
        N_out (int): Número de muestras (píxeles) en el plano de observación.
        radius (float): Radio de la apertura circular en el plano de entrada [m].
        lam (float): Longitud de onda de la luz [m].
        z (float): Distancia de propagación [m].

    Returns:
        numpy.ndarray: Matriz 2D con la intensidad del patrón de difracción.
    """
    print("Iniciando cálculo analítico de referencia...")
    start_time = time.time()

    # ---- 1. Configuración de las coordenadas del plano de salida ----
    dx_out = L_out / N_out
    x_out = np.linspace(-L_out / 2, L_out / 2, N_out)
    xx, yy = np.meshgrid(x_out, x_out)
    r = np.sqrt(xx**2 + yy**2)
    
    # ---- 2. Cálculo de la integral de difracción ----
    k = 2 * np.pi / lam
    r_flat = r.flatten()
    r_unique = np.unique(r_flat)
    U_radial = np.zeros_like(r_unique, dtype=np.complex128)

    print(f"Calculando la integral para {len(r_unique)} puntos radiales únicos...")

    def integrand(r_prime, r_obs):
        return r_prime * j0(k * r_obs * r_prime / z) * np.exp(1j * k * r_prime**2 / (2 * z))

    # Realiza la integración para cada punto radial único
    for i, r_obs in enumerate(r_unique):
        # Integramos la parte real y la parte imaginaria por separado.
        # Crucial: extraemos el PRIMER elemento ([0]) del resultado de quad.
        
        real_part = quad(lambda r_prime: integrand(r_prime, r_obs).real, 0, radius)[0]
        imag_part = quad(lambda r_prime: integrand(r_prime, r_obs).imag, 0, radius)[0]
        
        U_radial[i] = real_part + 1j * imag_part

    # ---- 3. Mapeo del resultado radial a la imagen 2D ----
    r_to_U_map = dict(zip(r_unique, U_radial))
    U_out_flat = np.array([r_to_U_map[val] for val in r_flat])
    U_out = U_out_flat.reshape((N_out, N_out))
    
    # ---- 4. Cálculo de la intensidad ----
    intensity = np.abs(U_out)**2
    if intensity.max() > 0:
        intensity = intensity / intensity.max()
        
    end_time = time.time()
    print(f"Cálculo completado en {end_time - start_time:.2f} segundos.")
    
    return intensity

def calcular_correlacion_pearson(imagen1, imagen2):
    """
    Calcula el coeficiente de correlación de Pearson entre dos imágenes 2D.

    Args:
        imagen1 (np.ndarray): La primera imagen (matriz 2D).
        imagen2 (np.ndarray): La segunda imagen (matriz 2D).

    Returns:
        float: El coeficiente de correlación de Pearson.
    """
    # 1. Asegurarse de que ambas imágenes tienen la misma forma
    if imagen1.shape != imagen2.shape:
        raise ValueError("Las imágenes deben tener las mismas dimensiones para la correlación.")

    # 2. Aplanar las matrices 2D para convertirlas en vectores 1D
    imagen1_plana = imagen1.flatten()
    imagen2_plana = imagen2.flatten()

    # 3. Calcular el coeficiente de correlación de Pearson
    # pearsonr devuelve el coeficiente y el p-valor, solo necesitamos el primero [0].
    coeficiente, _ = pearsonr(imagen1_plana, imagen2_plana)

    return coeficiente

# --- FIN DE LA IMPLEMENTACI�N DE FRESNEL FFT ---


if __name__ == "__main__":

 # =========================================================================
# PASO 1: PARÁMETROS DE ENTRADA PARA LA SIMULACIÓN FFT
# =========================================================================
# --- Parámetros Fundamentales ---
    WAVELENGTH = 633E-9      # Longitud de onda (633 nm)
    DISTANCE = 0.5           # Distancia de propagación z (0.5 m)
    APERTURE_RADIUS = 0.5e-3 # Radio de la apertura (0.5 mm)
    
    # --- Parámetros de la Malla de ENTRADA (Elegidos para ser VÁLIDOS) ---
    GRID_SIZE = 1024         # Número de píxeles (aumentamos para más detalle)
    PIXEL_PITCH_IN = 25E-6   # Tamaño del píxel de ENTRADA (25 µm)
    
    # --- Verificación de la Condición de Nyquist (Opcional pero recomendado) ---
    L_in = GRID_SIZE * PIXEL_PITCH_IN
    nyquist_check_value = GRID_SIZE * WAVELENGTH * DISTANCE
    if L_in**2 <= nyquist_check_value:
        print(f"✅ Condición de Nyquist CUMPLIDA: {L_in**2:.4e} <= {nyquist_check_value:.4e}")
    else:
        print(f"❌ ADVERTENCIA: Condición de Nyquist NO CUMPLIDA: {L_in**2:.4e} > {nyquist_check_value:.4e}")
    
    # --- Crear y Graficar el Campo de Entrada ---
    campo_entrada = OpticalField(GRID_SIZE, PIXEL_PITCH_IN, WAVELENGTH)
    campo_entrada.add_aperture("circ", size=APERTURE_RADIUS * 2) # La función espera diámetro
    campo_entrada.plot_intensity("Campo de Entrada Válido")
    
    # --- Ejecutar la Propagación FFT ---
    campo_salida = propagate_fresnel_fft(campo_entrada, DISTANCE)
    
    campo_salida.plot_intensity("Resultado transferencia de Fresnel")
    # =========================================================================
    # PASO 2: CONFIGURAR LA SIMULACIÓN ANALÍTICA CON LA SALIDA DE LA FFT
    # =========================================================================
    
    # --- Extraer los parámetros de la malla de SALIDA de la simulación FFT ---
    N_OUT_TARGET = campo_salida.size
    PIXEL_PITCH_OUT_TARGET = campo_salida.pixel_pitch
    L_OUT_TARGET = N_OUT_TARGET * PIXEL_PITCH_OUT_TARGET
    
    print("\n--- Parámetros para la Simulación Analítica (dictados por la FFT) ---")
    print(f"Píxeles de Salida (N_out): {N_OUT_TARGET}")
    print(f"Tamaño Físico de Salida (L_out): {L_OUT_TARGET * 1e3:.2f} mm")
    print(f"Pixel Pitch de Salida (dx_out): {PIXEL_PITCH_OUT_TARGET * 1e6:.2f} µm")
    
    # --- Ejecutar la Simulación Analítica ---
    analytical_intensity = calcular_difraccion_analitica_circular(
        L_out=L_OUT_TARGET,
        N_out=N_OUT_TARGET,
        radius=APERTURE_RADIUS,
        lam=WAVELENGTH,
        z=DISTANCE
    )
    
    # --- Graficar ambos resultados para comparar ---
    # (Usa el código de graficación que ya tienes, asegurando que ambos usen L_OUT_TARGET
    # para el parámetro 'extent' en imshow)
    
    
    plt.figure(figsize=(8, 7))
    plt.imshow(analytical_intensity, cmap='gray', extent=[-L_OUT_TARGET/2*1e3, L_OUT_TARGET/2*1e3, -L_OUT_TARGET/2*1e3, L_OUT_TARGET/2*1e3])
    plt.title("Resultado Analítico (Sincronizado)")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.colorbar(label="Intensidad Normalizada")
    plt.show()
    
    # ... (Después de haber calculado 'campo_salida' y 'analytical_intensity')
    
    # --- 1. Obtener la intensidad del campo FFT ---
    intensity_fft = np.abs(campo_salida.field)**2
    
    # --- 2. Normalizar ambas intensidades (buena práctica para visualización) ---
    if intensity_fft.max() > 0:
        intensity_fft_norm = intensity_fft / intensity_fft.max()
    if analytical_intensity.max() > 0:
        analytical_intensity_norm = analytical_intensity / analytical_intensity.max()
    
    # --- 3. Calcular la correlación entre los dos patrones de intensidad ---
    try:
        coeficiente = calcular_correlacion_pearson(intensity_fft_norm, analytical_intensity_norm)
        print("\n" + "="*50)
        print(f"📊 El coeficiente de correlación de Pearson es: {coeficiente:.6f}")
        print("="*50)
    except ValueError as e:
        print(f"Error al calcular la correlación: {e}")
    
    
    # --- 4. (Opcional pero recomendado) Visualizar un perfil central ---
    # Esto te permite ver las diferencias de forma muy clara
    centro = intensity_fft_norm.shape[0] // 2
    perfil_fft = intensity_fft_norm[centro, :]
    perfil_analitico = analytical_intensity_norm[centro, :]
    
    plt.figure(figsize=(12, 6))
    plt.title("Comparación de Perfil Central (Fila Central de la Imagen)")
    plt.plot(perfil_fft, label='Perfil FFT', linewidth=2)
    plt.plot(perfil_analitico, label='Perfil Analítico', linestyle='--', linewidth=2)
    plt.xlabel("Posición del Píxel")
    plt.ylabel("Intensidad Normalizada")
    plt.legend()
    plt.grid(True)
    plt.show()
