import numpy as np

# Sembollerin düzgün yazdırılması için numpy ayarı
np.set_printoptions(precision=8, suppress=True, floatmode='fixed')

print("--- 1. REFERANS KONFİGÜRASYONDAKİ KONUM (X) ---")
# 0.02 * X1^3 - 1.108 * X1 + 1.088 = 0 polinomunun köklerini bulalım
katsayilar = [0.02, 0, -1.108, 1.088]
kokler = np.roots(katsayilar)

# Kökleri ekrana yazdırıp ne olduklarını görelim
print("Bulunan tüm kökler:", kokler)

# İmajiner kısmı çok küçük olanları (gerçek kabul edip) filtreleyelim ve reel kısımlarını alalım
reel_kokler = np.real(kokler[np.abs(kokler.imag) < 1e-5])

# 0 ile 1 aralığında olanı seçelim (virgülden sonraki yuvarlama hatalarına karşı küçük bir pay bırakarak)
uygun_kokler = reel_kokler[(reel_kokler > -1e-5) & (reel_kokler < 1 + 1e-5)]

# X1 değerini atayalım ve tam sayıya yakınsıyorsa yuvarlayalım (analitik çözüm tam 1 çıkıyor)
X1 = float(np.round(uygun_kokler[0], 5))
X2 = 0.54 - 0.10 * X1**2
X3 = 0.0
X = np.array([X1, X2, X3])
print(f"Seçilen X = {X}\n")

print("--- 2. ŞEKİL DEĞİŞTİRME GRADYANI (F) ---")
t2 = np.pi / 4
sin_t2 = np.sin(t2)
cos_t2 = np.cos(t2)

# F matrisi elemanları (Kısmi türevler)
F11 = 1 + 0.20 * sin_t2 * X2
F12 = 0.20 * sin_t2 * X1
F21 = 0.20 * (1 - cos_t2) * X1
F22 = 1.0
F = np.array([
    [F11, F12, 0],
    [F21, F22, 0],
    [0,   0,   1]
])
print("F =\n", F, "\n")

print("--- 3. ŞEKİL DEĞİŞTİRME GRADYANININ TERSİ (F^-1) ---")
F_inv = np.linalg.inv(F)
print("F^-1 =\n", F_inv, "\n")

print("--- 4. GREEN ŞEKİL DEĞİŞTİRME TANSÖRÜ (C) ---")
C = np.dot(F.T, F)
print("C =\n", C, "\n")

print("--- 5. CAUCHY ŞEKİL DEĞİŞTİRME TANSÖRÜ (c) ---")
# Not: Klasik notasyonda Cauchy c = (F^-1)^T * F^-1 dir.
c = np.dot(F_inv.T, F_inv)
print("c =\n", c, "\n")

print("--- 6. PİOLA ŞEKİL DEĞİŞTİRME TANSÖRÜ (C^-1) ---")
C_inv = np.linalg.inv(C)
print("C^-1 =\n", C_inv, "\n")

print("--- 7. FINGER ŞEKİL DEĞİŞTİRME TANSÖRÜ (c^-1) ---")
# Finger tansörü B = F * F^T olarak da bilinir. 
c_inv = np.dot(F, F.T)
print("c^-1 =\n", c_inv, "\n")

print("--- 8. LAGRANGE GENLEME TANSÖRÜ (E) ---")
I = np.eye(3)
E_lagrange = 0.5 * (C - I)
print("E =\n", E_lagrange, "\n")

print("--- 9. EULER GENLEME TANSÖRÜ (e) ---")
e_euler = 0.5 * (I - c)
print("e =\n", e_euler, "\n")

print("--- 10. YER DEĞİŞTİRME (U) ---")
# U = x(t2) - X
# t2 anındaki x konumunu hesaplayalım
x1_t2 = X[0] + 0.20 * sin_t2 * X[0] * X[1]
x2_t2 = X[1] + 0.10 * (1 - cos_t2) * X[0]**2
x3_t2 = X[2]
x_t2 = np.array([x1_t2, x2_t2, x3_t2])
U = x_t2 - X
print("U =\n", U.reshape(3,1), "\n")

print("--- 11. YER DEĞİŞTİRME YARDIMIYLA LAGRANGE (E) ---")
# Yer değiştirme gradyanı (nabla U = F - I)
nabla_U = F - I
print("nabla_U =\n", nabla_U)
E_U = 0.5 * (nabla_U + nabla_U.T + np.dot(nabla_U.T, nabla_U))
print("E (U ile) =\n", E_U, "\n")

print("--- 12. LİNEERLEŞTİRİLMİŞ YAKLAŞIK LAGRANGE (E~) ---")
E_approx = 0.5 * (nabla_U + nabla_U.T)
print("E~ =\n", E_approx, "\n")

print("--- 13 & 14. GERME (Lambda) ve UZAMA ORANI (E_N) ---")
N1 = np.array([1, 0, 0])
N2 = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])

Lambda_N1 = np.sqrt(np.dot(N1.T, np.dot(C, N1)))
Lambda_N2 = np.sqrt(np.dot(N2.T, np.dot(C, N2)))

# Uzama oranı E_N = Lambda - 1
E_N1 = Lambda_N1 - 1
E_N2 = Lambda_N2 - 1

print(f"Lambda_N1 = {Lambda_N1:.8f}, E_N1 = {E_N1:.8f}")
print(f"Lambda_N2 = {Lambda_N2:.8f}, E_N2 = {E_N2:.8f}\n")

print("--- 15. AÇI DEĞİŞİMİ ---")
# Başlangıç açısı
theta_0 = np.arccos(np.dot(N1, N2))
print(f"Başlangıç açısı (theta_0) = {np.degrees(theta_0):.8f} derece")

# Şekil değiştirme sonrası doğrultular
n1 = np.dot(F, N1) / Lambda_N1
n2 = np.dot(F, N2) / Lambda_N2

# Yeni açı
theta = np.arccos(np.dot(n1, n2))
delta_theta_rad = theta - theta_0

print(f"Yeni açı (theta) = {np.degrees(theta):.8f} derece")
print(f"Açı değişimi (Delta theta) = {np.degrees(delta_theta_rad):.8f} derece")
print(f"Açı değişimi (Delta theta) = {delta_theta_rad:.8f} rad")