import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, freqz, lfilter, impulse

# Paramètres du filtre
fc1 = 1e3  # Fréquence de coupure basse (Hz)
fc2 = 3e3  # Fréquence de coupure haute (Hz)
fe = 8e3   # Fréquence d'échantillonnage (Hz)
order = 4  # Ordre du filtre

# Fréquences normalisées
nyquist = fe / 2
fc1_n = fc1 / nyquist
fc2_n = fc2 / nyquist

# Conception du filtre Butterworth
b, a = butter(order, [fc1_n, fc2_n], btype='band')

# 1. Coefficients du filtre
print("Coefficients b:", b)
print("Coefficients a:", a)

# 2. Réponse impulsionnelle h(n)
n_samples = 50
impulse_signal = np.zeros(n_samples)
impulse_signal[0] = 1  # Impulsion de Dirac
h = lfilter(b, a, impulse_signal)

# Tracé de la réponse impulsionnelle
plt.figure(figsize=(10, 4))
plt.stem(h, use_line_collection=True)
plt.title("Réponse impulsionnelle $h(n)$")
plt.xlabel("n (échantillons)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 3. Fonction de transfert H(f)
w, h_freq = freqz(b, a, worN=1024, fs=fe)
plt.figure(figsize=(10, 4))
plt.plot(w, 20 * np.log10(abs(h_freq)))
plt.title("Réponse en fréquence $H(f)$")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.grid()
plt.show()

# 4. Effet de l'ordre du filtre
orders = [2, 4, 8]
plt.figure(figsize=(10, 6))
for o in orders:
    b_o, a_o = butter(o, [fc1_n, fc2_n], btype='band')
    _, h_freq_o = freqz(b_o, a_o, worN=1024, fs=fe)
    plt.plot(w, 20 * np.log10(abs(h_freq_o)), label=f"Ordre {o}")

plt.title("Effet de l'ordre sur la réponse en fréquence")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude (dB)")
plt.legend()
plt.grid()
plt.show()
