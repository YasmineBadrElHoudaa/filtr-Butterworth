import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Paramètres donnés
fc1 = 1000  # Fréquence de coupure inférieure en Hz
fc2 = 3000  # Fréquence de coupure supérieure en Hz
fe = 8000   # Fréquence d'échantillonnage en Hz

# 1. Type de filtre
# Il s'agit d'un filtre passe-bande (PB), car les deux fréquences de coupure définissent une bande.

# 2. Calcul des fréquences de coupure normalisées
Wn1 = fc1 / (fe / 2)  # Normalisation de la fréquence fc1
Wn2 = fc2 / (fe / 2)  # Normalisation de la fréquence fc2
print("Fréquences de coupure normalisées : Wn1 =", Wn1, ", Wn2 =", Wn2)

# 3. Calcul des coefficients du filtre Butterworth (passe-bande)
ordre = 4  # Ordre du filtre
b, a = signal.butter(ordre, [Wn1, Wn2], btype='bandpass', analog=False)

# Affichage des coefficients b et a
print("Coefficients b :", b)
print("Coefficients a :", a)

# 4. Tracer la réponse impulsionnelle h(n)
system = signal.dlti(b, a)  # Définir le système

# Réponse impulsionnelle
t, h = signal.dimpulse(system)

# Extraire la réponse de la forme de liste
h = np.squeeze(h)

# affichage de  la réponse impulsionnelle
plt.stem(t, h, basefmt=" ")
plt.title("Réponse impulsionnelle h(n)")
plt.xlabel("n")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 5. la fonction de transfert H(f)
frequencies = np.linspace(0, fe / 2, 500)  # Plage de fréquences
w, h_freq = signal.freqz(b, a, worN=frequencies, fs=fe)  # Fonction de transfert

plt.plot(w, 20 * np.log10(np.abs(h_freq)), 'b')
plt.title("Fonction de transfert H(f)")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Gain (dB)")
plt.grid()
plt.show()

# 6. Effet de l'augmentation de l'ordre du filtre : essayons  avec un ordre =8 et vérifions la fonction de transfert pour cet ordre par rapport a l'ordre 4
ordre_haut = 8  # Augmenter l'ordre
b_haut, a_haut = signal.butter(ordre_haut, [Wn1, Wn2], btype='bandpass', analog=False)
w_haut, h_haut = signal.freqz(b_haut, a_haut, worN=frequencies, fs=fe)  # Nouvelle fonction de transfert

plt.plot(w_haut, 20 * np.log10(np.abs(h_haut)), 'r', label="Ordre 8")
plt.plot(w, 20 * np.log10(np.abs(h_freq)), 'b', label="Ordre 4")
plt.title("Fonction de transfert H(f) avec ordre variable")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Gain (dB)")
plt.legend()
plt.grid()
plt.show()
