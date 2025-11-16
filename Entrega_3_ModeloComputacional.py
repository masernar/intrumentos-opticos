import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk             # <-- Importamos tkinter
from tkinter import filedialog   # <-- Importamos el diálogo de archivos

# --- 0. Configuración de Precisión ---
dtype_complejo = np.complex128

# --- BLOQUE DE SELECCIÓN DE ARCHIVO (LOCAL / VS CODE) ---
# Creamos una ventana raíz de tkinter (necesaria) y la ocultamos
root = tk.Tk()
root.withdraw()

print("Abriendo el explorador de archivos para seleccionar imagen...")

# Abrimos el diálogo para "Abrir Archivo" y guardamos la ruta
nombre_archivo_objeto = filedialog.askopenfilename(
    title="Selecciona tu imagen de objeto",  # <-- ¡CORREGIDO! Este es solo el título de la ventana
    filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg"),
               ("Todos los archivos", "*.*")]
)

if nombre_archivo_objeto:  # Si el usuario seleccionó un archivo
    print(f"¡Éxito! Se seleccionó el archivo:")
    print(nombre_archivo_objeto)  # Imprime la ruta completa
else:  # Si el usuario hizo clic en "Cancelar"
    print("No se seleccionó ningún archivo.")

print("-" * 30)

# --- 1. Parámetros Físicos (Unidades en mm) ---
print("Iniciando simulación de microscopio 4f (Punto 2)...")
LAMBDA = 533e-6   # 533 nm convertidos a milímetros
NA = 0.5         # Apertura Numérica del objetivo
M_OBJETIVO = 20.0  # Aumento del objetivo
F_TL = 200.0       # Distancia focal Lente Tubo (mm)

# Parámetros deducidos
F_MO = F_TL / M_OBJETIVO
R_PUPILA = NA * F_MO

print(f"Parámetros: λ={LAMBDA*1000:.0f}nm, f_MO={F_MO:.1f}mm, R_Pupila={R_PUPILA:.1f}mm")

# --- 2. Malla de Simulación (Plano Pupila P(x,y)) ---
N = 1024  # Número de píxeles (Resolución de la simulación). 1024x1024
L_PUPILA = 20.0  # Ancho total de la "ventana" de simulación (mm)
dx = L_PUPILA / N # Tamaño de píxel en plano pupila (mm)
eje_x = np.linspace(-L_PUPILA/2, L_PUPILA/2 - dx, N)
X, Y = np.meshgrid(eje_x, eje_x)

# --- 3. Malla y Objeto (Plano Objeto S(ξ,η)) ---
# Escala del plano objeto (conjugado de Fourier)
d_xi = (LAMBDA * F_MO) / L_PUPILA
d_xi_micras = d_xi * 1000  # Convertir a micras (µm)
print(f"Tamaño píxel ENTRADA (Δξ): {d_xi_micras:.3f} µm/píxel")

L_OBJETO_MICRAS = d_xi_micras * N
eje_xi_micras = np.linspace(-L_OBJETO_MICRAS/2, L_OBJETO_MICRAS/2 - d_xi_micras, N)

# --- (NUEVO) CÁLCULO DE EJES DE SALIDA (Magnificación) ---
# La escala de la segunda TF depende de f_TL
d_u = (LAMBDA * F_TL) / L_PUPILA
d_u_micras = d_u * 1000
print(f"Tamaño píxel SALIDA (Δu): {d_u_micras:.3f} µm/píxel (Magnificación de {d_u/d_xi:.0f}x)")

L_IMAGEN_MICRAS = d_u_micras * N
eje_u_micras = np.linspace(-L_IMAGEN_MICRAS/2, L_IMAGEN_MICRAS/2 - d_u_micras, N)


# --- Creación del Objeto S(ξ, η) ---
if nombre_archivo_objeto:
    # Si el usuario subió un archivo, lo usamos
    try:
        print(f"Cargando '{nombre_archivo_objeto}' como objeto...")
        img = Image.open(nombre_archivo_objeto).convert('L').resize((N, N))
        S_amplitud = np.array(img) / 255.0
        S_muestra = S_amplitud.astype(dtype_complejo)
        print(f"Objeto cargado exitosamente.")
    except Exception as e:
        print(f"Error al leer la imagen: {e}. Usando impulso delta.")
        nombre_archivo_objeto = None # Forzar el plan B
else:
    print("No se cargó un archivo.")

if not nombre_archivo_objeto:
    # Si no hay archivo, usamos el impulso delta
    print("...Usando un impulso delta (punto) en su lugar.")
    S_muestra = np.zeros((N, N), dtype=dtype_complejo)
    S_muestra[N//2, N//2] = 1.0 + 0.0j


# --- 4. Creación de la Función Pupila P(x,y) ---
R = np.sqrt(X**2 + Y**2)
P_pupila = np.zeros((N, N), dtype=dtype_complejo)
P_pupila[R <= R_PUPILA] = 1.0 + 0.0j

R_STOP_PORCENTAJE = 0.2
R_STOP = R_STOP_PORCENTAJE * R_PUPILA

P_stop = np.zeros((N, N), dtype=dtype_complejo)
P_stop[R <= R_STOP] = 1.0 + 0.0j

P_pupila = P_pupila - P_stop
    
# --- 5. Simulación 4f (La Doble Transformada) ---
print("Ejecutando modelo 4f: FFT(P * FFT(S))...")

def tf_optica(campo):
    # Esta función usa fft2 (Transformada Directa)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(campo)))

# Etapa 1: Objeto -> Pupila (TF 1)
U_pupila = tf_optica(S_muestra)

# Etapa 2: Filtrado en la Pupila (multiplicación)
U_pupila_filtrada = U_pupila * P_pupila

# Etapa 3: Pupila -> Imagen (TF 2)
# ¡¡CORRECCIÓN FÍSICA!!
# La segunda lente también hace una Transformada DIRECTA (fft).
U_imagen = tf_optica(U_pupila_filtrada)

# Etapa 4: Detección en la Cámara (Intensidad)
I_imagen_simulada = np.abs(U_imagen)**2

if np.max(I_imagen_simulada) > 0:
    I_imagen_simulada /= np.max(I_imagen_simulada)

print("¡Simulación completada!")

# --- 6. Visualización y Análisis ---
r_abbe = (LAMBDA * 1000) / (2 * NA)  # 0.533 µm
print(f"\nLímite de Abbe (r = λ/2NA): {r_abbe:.3f} µm")

# --- Gráficos ---
plt.figure(figsize=(18, 6))

# 1. Objeto S(ξ, η)
plt.subplot(1, 3, 1)
plt.imshow(np.abs(S_muestra), cmap='gray',
           extent=[eje_xi_micras.min(), eje_xi_micras.max(), eje_xi_micras.min(), eje_xi_micras.max()])
plt.title(f'Objeto de Entrada S(ξ, η)')
plt.xlabel(f'ξ (µm) [Ancho total: {L_OBJETO_MICRAS:.0f} µm]')
plt.ylabel('η (µm)')

# 2. Pupila P(x, y)
plt.subplot(1, 3, 2)
plt.imshow(np.abs(P_pupila), cmap='gray',
           extent=[eje_x.min(), eje_x.max(), eje_x.min(), eje_x.max()])
plt.title(f'Pupila P(x, y) (NA = {NA})')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')

# 3. Imagen (Intensidad)
plt.subplot(1, 3, 3)
plt.imshow(I_imagen_simulada, cmap='gray',
           # ¡¡CORRECCIÓN DE MAGNIFICACIÓN!!
           # Usamos los ejes de salida 'eje_u_micras'
           extent=[eje_u_micras.min(), eje_u_micras.max(), eje_u_micras.min(), eje_u_micras.max()])
plt.title('Imagen Simulada |U_img|²')
plt.xlabel(f'u (µm) [Ancho total: {L_IMAGEN_MICRAS:.0f} µm]')
plt.ylabel('v (µm)')

plt.tight_layout()
plt.show()