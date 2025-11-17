import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog
import os
import io

# --- 0. Configuración de Precisión ---
dtype_complejo = np.complex128

# --- BLOQUE DE SELECCIÓN DE ARCHIVO (LOCAL / VS CODE) ---
root = tk.Tk()
root.withdraw()

print("Abriendo el explorador de archivos para seleccionar imagen...")
nombre_archivo_objeto = filedialog.askopenfilename(
    title="Selecciona tu imagen o archivo de datos",
    filetypes=[
        ("Imágenes", "*.png;*.jpg;*.jpeg"),
        ("Datos CSV", "*.csv"),
        ("Datos DAT", "*.dat"),
        ("Todos los archivos", "*.*")
    ]
)

if nombre_archivo_objeto:
    print(f"¡Éxito! Se seleccionó el archivo:")
    print(nombre_archivo_objeto)
else:
    print("No se seleccionó ningún archivo.")

print("-" * 30)

# --- 1. Parámetros Físicos (Unidades en mm) ---
print("Iniciando simulación de microscopio 4f...")
LAMBDA = 533e-6   # 533 nm convertidos a milímetros
NA = 0.5         # Apertura Numérica del objetivo
M_OBJETIVO = 20.0  # Aumento del objetivo
F_TL = 200.0       # Distancia focal Lente Tubo (mm)

# Parámetros deducidos
F_MO = F_TL / M_OBJETIVO
R_PUPILA = NA * F_MO

print(f"Parámetros: λ={LAMBDA*1000:.0f}nm, f_MO={F_MO:.1f}mm, R_Pupila={R_PUPILA:.1f}mm")

# --- 2. Malla de Simulación (Plano Pupila P(x,y)) ---
N = 1024
L_PUPILA = 20.0
dx = L_PUPILA / N
eje_x = np.linspace(-L_PUPILA/2, L_PUPILA/2 - dx, N)
X, Y = np.meshgrid(eje_x, eje_x)

# --- 3. Malla y Objeto (Plano Objeto S(ξ,η)) ---
# Escala del plano objeto (conjugado de Fourier)
d_xi = (LAMBDA * F_MO) / L_PUPILA
d_xi_micras = d_xi * 1000
print(f"Tamaño píxel ENTRADA (Δξ): {d_xi_micras:.3f} µm/píxel")

L_OBJETO_MICRAS = d_xi_micras * N
eje_xi_micras = np.linspace(-L_OBJETO_MICRAS/2, L_OBJETO_MICRAS/2 - d_xi_micras, N)

# --- CÁLCULO DE EJES DE SALIDA (Magnificación) ---
d_u = (LAMBDA * F_TL) / L_PUPILA
d_u_micras = d_u * 1000
print(f"Tamaño píxel SALIDA (Δu): {d_u_micras:.3f} µm/píxel (Magnificación de {d_u/d_xi:.0f}x)")

L_IMAGEN_MICRAS = d_u_micras * N
eje_u_micras = np.linspace(-L_IMAGEN_MICRAS/2, L_IMAGEN_MICRAS/2 - d_u_micras, N)


# --- Creación del Objeto S(ξ, η) ---
#    (LÓGICA MEJORADA PARA CAMPO COMPLEJO con 'i' -> 'j')

S_muestra_cargada = False  # Bandera para saber si cargamos algo

if nombre_archivo_objeto:
    try:
        # --- NUEVO: Comprobar la extensión del archivo ---
        _, extension = os.path.splitext(nombre_archivo_objeto)
        print(f"Detectada extensión: {extension}")

        if extension.lower() in ['.png', '.jpg', '.jpeg']:
            # --- MÉTODO PARA IMÁGENES DE AMPLITUD (Como antes) ---
            print(f"Cargando '{nombre_archivo_objeto}' como imagen de Amplitud...")
            img = Image.open(nombre_archivo_objeto).convert('L')
            img_resized = img.resize((N, N), Image.Resampling.LANCZOS)
            S_amplitud = np.array(img_resized) / 255.0
            S_muestra = S_amplitud.astype(dtype_complejo)  # Solo Amplitud, Fase=0

        elif extension.lower() in ['.csv', '.dat']:
            # --- NUEVO MÉTODO (para CAMPO COMPLEJO con 'i') ---
            print(f"Cargando '{nombre_archivo_objeto}' como CAMPO COMPLEJO...")
            
            # --- INICIO DE LA CORRECCIÓN i -> j ---
            with open(nombre_archivo_objeto, 'r') as f:
                raw_text = f.read()
            
            print("Traduciendo 'i' (científico) a 'j' (Python)...")
            processed_text = raw_text.replace('i', 'j')
            
            text_stream = io.StringIO(processed_text)
            # --- FIN DE LA CORRECCIÓN ---

            # Cargar los datos desde el stream de texto con np.loadtxt
            try:
                S_complex_data = np.loadtxt(text_stream, delimiter=',', dtype=np.complex128)
            except ValueError:
                print("Fallo al leer con comas, reintentando con espacios...")
                text_stream.seek(0) 
                S_complex_data = np.loadtxt(text_stream, dtype=np.complex128)

            print(f"Datos complejos cargados. Dimensiones originales: {S_complex_data.shape}")

            # Redimensionar el campo (Parte Real e Imaginaria por separado)
            S_real = S_complex_data.real
            img_real = Image.fromarray(S_real.astype(np.float32), mode='F')
            img_real_resized = img_real.resize((N, N), Image.Resampling.LANCZOS)
            S_real_final = np.array(img_real_resized)
            
            S_imag = S_complex_data.imag
            img_imag = Image.fromarray(S_imag.astype(np.float32), mode='F')
            img_imag_resized = img_imag.resize((N, N), Image.Resampling.LANCZOS)
            S_imag_final = np.array(img_imag_resized)
            
            # Recombinar
            S_muestra = S_real_final + 1j * S_imag_final
            print(f"Campo complejo redimensionado a ({N}, {N})")

        else:
            print(f"Extensión '{extension}' no reconocida. Usando impulso delta.")
            raise ValueError("Formato no soportado")

        S_muestra_cargada = True

    except Exception as e:
        print(f"Error al leer el archivo: {e}. Usando impulso delta.")

else:
    print("No se cargó un archivo.")

# --- Plan B (Impulso delta) ---
if not S_muestra_cargada:
    print("...Usando un impulso delta (punto) en su lugar.")
    S_muestra = np.zeros((N, N), dtype=dtype_complejo)
    S_muestra[N//2, N//2] = 1.0 + 0.0j

# --- FIN DE LA SECCIÓN 3 ---


# --- 4. Creación de la Función Pupila P(x,y) ---
print("-" * 30)
R = np.sqrt(X**2 + Y**2)

# 4.1. Pupila de Campo Claro (Bright Field - BF)
P_claro = np.zeros((N, N), dtype=dtype_complejo)
P_claro[R <= R_PUPILA] = 1.0 + 0.0j

# 4.2. Pupila de Campo Oscuro (Dark Field - DF)
R_STOP_PORCENTAJE = 0.001
R_STOP = R_STOP_PORCENTAJE * R_PUPILA
print(f"Radio del Stop (DF y Fase): {R_STOP:.2f} mm")

# Creamos el stop
P_stop = np.zeros((N, N), dtype=dtype_complejo)
P_stop[R <= R_STOP] = 1.0 + 0.0j

# La pupila DF es la pupila BF MENOS el stop central
P_oscuro = P_claro - P_stop

# 4.3. Pupila de Contraste de Fase (Zernike)
AMPLITUD_PLATO = 0.8 # 30% de amplitud
FASE_PLATO = 1j      # Desfase de pi/2 (90 grados)

P_fase = P_claro.copy()
mascara_plato_fase = (R <= R_STOP)
P_fase[mascara_plato_fase] = AMPLITUD_PLATO * FASE_PLATO

# 4.4. Selección de modo
modo_elegido = input("Elige el modo de simulación:\n 1. Campo Claro (Default)\n 2. Campo Oscuro\n 3. Contraste de Fase\nSelecciona (1, 2 o 3): ")

if modo_elegido == "2":
    P_pupila_final = P_oscuro
    titulo_modo = "Campo Oscuro (HPF)"
    print("¡Modo CAMPO OSCURO seleccionado!")
elif modo_elegido == "3":
    P_pupila_final = P_fase
    titulo_modo = "Contraste de Fase (Zernike)"
    print("¡Modo CONTRASTE DE FASE seleccionado!")
else:
    P_pupila_final = P_claro
    titulo_modo = "Campo Claro (BF)"
    print("¡Modo CAMPO CLARO seleccionado!")


# --- 5. Simulación 4f (La Doble Transformada) ---
print("Ejecutando modelo 4f: FFT(P * FFT(S))...")

def tf_optica(campo):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(campo)))

# Etapa 1: Objeto -> Pupila (TF 1)
U_pupila = tf_optica(S_muestra)

# Etapa 2: Filtrado en la Pupila (multiplicación)
U_pupila_filtrada = U_pupila * P_pupila_final

# Etapa 3: Pupila -> Imagen (TF 2)
U_imagen = tf_optica(U_pupila_filtrada)

# Etapa 4: Detección en la Cámara (Intensidad)
I_imagen_simulada = np.abs(U_imagen)**2

if np.max(I_imagen_simulada) > 0:
    I_imagen_simulada /= np.max(I_imagen_simulada)

print("¡Simulación completada!")

# --- 6. Visualización y Análisis ---
#    (¡¡SECCIÓN COMPLETAMENTE REEMPLAZADA!!)

print(f"\nGenerando {5 if S_muestra_cargada else 4} gráficos separados...")

# --- Gráfico 1: Objeto S(ξ, η) - Amplitud ---
S_amplitud = np.abs(S_muestra)
plt.figure(figsize=(8, 7))
im1 = plt.imshow(S_amplitud, cmap='gray',
           extent=[eje_xi_micras.min(), eje_xi_micras.max(), eje_xi_micras.min(), eje_xi_micras.max()])
plt.title(f'Objeto S(ξ, η) - AMPLITUD')
plt.xlabel(f'ξ (µm) [Ancho total: {L_OBJETO_MICRAS:.0f} µm]')
plt.ylabel('η (µm)')
plt.colorbar(im1, label='Amplitud (Unidades arbitrarias)')


# --- Gráfico 2: Objeto S(ξ, η) - Fase ---
S_fase = np.angle(S_muestra)
plt.figure(figsize=(8, 7))
im2 = plt.imshow(S_fase, cmap='twilight_shifted',
           extent=[eje_xi_micras.min(), eje_xi_micras.max(), eje_xi_micras.min(), eje_xi_micras.max()])
plt.title(f'Objeto S(ξ, η) - FASE')
plt.xlabel(f'ξ (µm)')
plt.ylabel('η (µm)')
plt.colorbar(im2, label='Fase (Radianes)')


# --- Gráfico 3: Pupila P(x, y) - Amplitud ---
# (Importante para ver el stop de Zernike)
P_amplitud = np.abs(P_pupila_final)
plt.figure(figsize=(8, 7))
im3 = plt.imshow(P_amplitud, cmap='gray',
           extent=[eje_x.min(), eje_x.max(), eje_x.min(), eje_x.max()],
           vmin=0, vmax=1.0) # Forzamos la escala 0-1
plt.title(f'Pupila P(x, y) - AMPLITUD - Modo: {titulo_modo}')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar(im3, label='Transmitancia (Amplitud)')


# --- Gráfico 4: Pupila P(x, y) - Fase ---
# (El gráfico clave para el Contraste de Fase)
P_fase = np.angle(P_pupila_final)
plt.figure(figsize=(8, 7))
im4 = plt.imshow(P_fase, cmap='twilight_shifted',
           extent=[eje_x.min(), eje_x.max(), eje_x.min(), eje_x.max()])
plt.title(f'Pupila P(x, y) - FASE - Modo: {titulo_modo}')
plt.xlabel('x (mm)')
plt.ylabel('y (mm)')
plt.colorbar(im4, label='Fase (Radianes)')


# --- Gráfico 5: Imagen Final (Intensidad) ---
plt.figure(figsize=(8, 7))
# Usamos vmin/vmax para que la escala sea consistente si no normalizamos
im5 = plt.imshow(I_imagen_simulada, cmap='gray',
           extent=[eje_u_micras.min(), eje_u_micras.max(), eje_u_micras.min(), eje_u_micras.max()],
           vmin=0, vmax=1.0)
plt.title(f'Imagen de la muestra Simulada - Modo: {titulo_modo}')
plt.xlabel(f'u (µm) [Ancho total: {L_IMAGEN_MICRAS:.0f} µm]')
plt.ylabel('v (µm)')
plt.colorbar(im5, label='Intensidad Normalizada')


# --- Gráfico 6: Corte Transversal ---
plt.figure(figsize=(10, 5))
if not S_muestra_cargada:
    # Mostramos un corte transversal de la PSF
    perfil_psf = I_imagen_simulada[N//2, :]
    plt.plot(eje_u_micras, perfil_psf)
    plt.title(f'PSF (Corte transversal) - Modo: {titulo_modo}')
    plt.xlabel('u (µm)')
    plt.ylabel('Intensidad')
    plt.grid(True)
else:
    # Si cargamos una muestra, un corte de la imagen es más útil
    perfil_imagen = I_imagen_simulada[N//2, :]
    plt.plot(eje_u_micras, perfil_imagen)
    plt.title(f'Imagen (Corte transversal central) - Modo: {titulo_modo}')
    plt.xlabel('u (µm)')
    plt.ylabel('Intensidad Normalizada')
    plt.grid(True)
    
plt.figure(figsize=(8, 5))
plt.hist(S_fase.flatten(), bins=100, range=(-0.1, 1.0)) # Rango (0, 1)
plt.title('Histograma de Valores de Fase de la Muestra')
plt.xlabel('Corrimiento de Fase (Radianes)')
plt.ylabel('Conteo de Píxeles')
plt.grid(True)
plt.show()
# --- Mostrar todos los gráficos ---
plt.show()