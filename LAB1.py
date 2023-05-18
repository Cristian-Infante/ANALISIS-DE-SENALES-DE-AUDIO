import librosa                      # Librería para cargar y manipular archivos de audio
import numpy as np                  # Librería para manejo de arreglos y matrices, y cálculo numérico
import matplotlib.pyplot as plt     # Librería para graficar datos

# Cargar el audio
#filename = 'c:/Users/ASUS/OneDrive/Escritorio/PROCESOS ESTOCASTICOS/LABS/Grabación.wav'
filename = 'c:/Users/ASUS/OneDrive/Escritorio/PROCESOS ESTOCASTICOS/LABS/blanco.wav'

# mono=True y dtype=np.float32 para que se compatible con la transformada de Fourier.
audio, fz = librosa.load(filename, mono=True, dtype=np.float32)                                                     # fz es la frecuencia de muestreo y audio es el arreglo de muestras de audio



# Definir l_salto y audio_d
l_ventana = 0.03                                                                                                    # ventana de 30 ms para calcular la transformada de Fourier
l_salto = int(fz * l_ventana / 2)
audio_d = len(audio) / fz

# Información básica del archivo
print("Duración del archivo de audio:", audio_d, "segundos")
print("Frecuencia de muestreo:", fz, "Hz")



# Calcular la transformada de Fourier
fft_signal = np.fft.fft(audio)
fft_norm = np.abs(fft_signal)

# Verificar si la señal es constante
tolerance = 0.001
constant_time = 0
for i in range(1, len(fft_norm)):
    if np.abs(fft_norm[i] - fft_norm[i-1]) > tolerance:
        constant_time = 0
    else:
        constant_time += l_salto / fz
    if constant_time > 0.5 * audio_d:
        is_constant = True
        break
else:
    is_constant = False

# Graficar la norma de la transformada de Fourier
freqs = np.fft.fftfreq(len(audio)) * fz
fig, ax = plt.subplots()
ax.set_title('Norma de la Transformada de Fourier')
ax.plot(freqs[:len(freqs)//2], fft_norm[:len(fft_norm)//2])
ax.set(xlabel='Frequency (Hz)', ylabel='Magnitud', title='Norma de la Transformada de Fourier')

# Agregar el texto de verificación de la gráfica de la norma
if is_constant:
    plt.text(0.5, 0.95, "La señal es constante", transform=plt.gca().transAxes)
else:
    plt.text(0.5, 0.95, "La señal no es constante", transform=plt.gca().transAxes)



# Calcular la autocovarianza
cov = np.correlate(audio, audio, mode='full')
lags = librosa.times_like(cov, sr=fz)

# Graficar la autocovarianza
fig, ax = plt.subplots()
ax.set_title('Autocovarianza')
ax.plot(lags, cov)
ax.set(xlabel='Tiempo (s)', ylabel='Magnitud', title='Autocovarianza')



# Calcular la autocorrelación
corr = librosa.autocorrelate(audio, max_size=len(audio)//2)
lags = librosa.times_like(corr, sr=fz) 

# Graficar la autocorrelación
fig, ax = plt.subplots()
ax.set_title('Autocorrelación')
ax.plot(lags, corr)
ax.set(xlabel='Tiempo (s)', ylabel='Magnitud', title='Autocorrelación')



#calular forma de onda
onda = librosa.stft(audio, hop_length=l_salto, win_length=l_salto)

# Gráfico de la forma de onda
fig, ax = plt.subplots()
ax.set_title('Forma de onda')
ax.plot(onda)
ax.set(xlabel='Tiempo (s)', ylabel='Amplitud', title='Forma de onda')



# Mostrar todas las gráficas
plt.show()
