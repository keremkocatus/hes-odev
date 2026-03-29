"""
MAT374 – Hesaplamalı Sürekli Ortamlar Mekaniği ÖDEV 1 


Hareket:
    x1 = X1 + 0.20 * sin(t) * X1 * X2
    x2 = X2 + 0.10 * (1 - cos(t)) * X1^2
    x3 = X3

Kısım I  : ChatGPT çözümünü doğrular (t1=π/2, x=(1.088,0.54,0), t2=π/4, N1=[1,0,0], N2=[1/√2,1/√2,0])
Kısım II : Kendi seçimlerimizle sıfırdan her şeyi hesaplar

Her sonuç referansa (Kısım I) karşı karşılaştırılır veya sıfırdan hesaplanır (Kısım II),
her büyüklüğün yanında açık bir  ✓ DOĞRU  /  ✗ YANLIŞ  etiketi bulunur.
"""

import numpy as np
from numpy.linalg import inv, norm

np.set_printoptions(precision=8, suppress=True, floatmode="fixed")

# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı Fonksiyonlar
# ─────────────────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"

def header(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'═'*64}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═'*64}{RESET}")

def section(title: str) -> None:
    print(f"\n{BOLD}── {title} ──{RESET}")

def check(label: str, computed, reference=None, tol: float = 1e-5) -> None:
    """
    Hesaplanan değeri yazdırır.
    Referans verilmişse ✓ DOĞRU veya ✗ YANLIŞ yazdırır.
    Referans skaler, 1-B dizi veya 2-B dizi olabilir.
    """
    comp = np.asarray(computed, dtype=float)

    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        ok  = np.allclose(comp, ref, atol=tol)
        tag = f"{GREEN}✓ CORRECT{RESET}" if ok else f"{RED}✗ WRONG{RESET}"
        print(f"{label}:\n{comp}\n  → {tag}  (reference: {ref})")
    else:
        print(f"{label}:\n{comp}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Temel Hesaplama Motoru
# ─────────────────────────────────────────────────────────────────────────────

def find_reference_position(x_current: np.ndarray, t1: float, tol: float = 1e-10) -> np.ndarray:
    """
    t1 anında verilen uzaysal konum x'ten, hareket denklemlerini sayısal olarak
    ters çevirerek referans (malzeme) konumu X = (X1, X2, X3) bulunur.

    Hareket:
        x1 = X1 + 0.20*sin(t1)*X1*X2   =>  x1 = X1*(1 + 0.20*sin(t1)*X2)
        x2 = X2 + 0.10*(1-cos(t1))*X1^2

    Strateji:
        x2'den: X2 = x2 - 0.10*(1-cos(t1))*X1^2
        x1 denklemine yerleştir → X1 cinsinden kübik → np.roots ile çöz.
        Gerçek olan ve doğru eşleşen kökü seç.
    """
    s1 = np.sin(t1)
    c1 = np.cos(t1)
    x1_given, x2_given, x3_given = x_current

    # Kübiği oluştur: 0.02*sin(t1)*X1^3 - (1 + 0.20*sin(t1)*x2_given)*X1 + x1_given ... bekle,
    # düzgün türetelim.
    #
    # X2 = x2_given - 0.10*(1-c1)*X1^2
    # x1_given = X1*(1 + 0.20*s1*(x2_given - 0.10*(1-c1)*X1^2))
    #          = X1 + 0.20*s1*x2_given*X1 - 0.02*s1*(1-c1)*X1^3
    #
    # => 0.02*s1*(1-c1)*X1^3 - (1 + 0.20*s1*x2_given)*X1 + x1_given = 0

    a3 = 0.02 * s1 * (1 - c1)
    a1 = -(1.0 + 0.20 * s1 * x2_given)
    a0 = x1_given

    if abs(a3) < 1e-14:
        # Dejenere durum (örn. t1 = 0 veya t1 = π): doğrusal denklem
        X1 = -a0 / a1
        roots_real = np.array([X1])
    else:
        coeffs = [a3, 0.0, a1, a0]
        roots  = np.roots(coeffs)
        # Sadece reel kökleri tut
        roots_real = np.real(roots[np.abs(roots.imag) < 1e-6])

    # Reel kökler arasından x'i en doğru şekilde yeniden üreten kökü seç
    best_X1, best_err = None, np.inf
    for r in roots_real:
        X2_candidate = x2_given - 0.10 * (1 - c1) * r**2
        x1_back = r + 0.20 * s1 * r * X2_candidate
        x2_back = X2_candidate + 0.10 * (1 - c1) * r**2
        err = abs(x1_back - x1_given) + abs(x2_back - x2_given)
        if err < best_err:
            best_err, best_X1 = err, r

    X1 = float(best_X1)
    X2 = float(x2_given - 0.10 * (1 - c1) * X1**2)
    X3 = float(x3_given)
    return np.array([X1, X2, X3])


def deformation_gradient(X: np.ndarray, t: float) -> np.ndarray:
    """
    Malzeme noktası X ve zaman t'de deformasyon gradyanı F = dx/dX hesaplar.

    Kısmi türevler:
        F11 = ∂x1/∂X1 = 1 + 0.20*sin(t)*X2
        F12 = ∂x1/∂X2 = 0.20*sin(t)*X1
        F13 = 0
        F21 = ∂x2/∂X1 = 0.20*(1-cos(t))*X1      [Not: 2*0.10 = 0.20]
        F22 = ∂x2/∂X2 = 1
        F23 = 0
        F31 = F32 = 0,  F33 = 1
    """
    X1, X2 = X[0], X[1]
    s = np.sin(t)
    c = np.cos(t)

    F = np.array([
        [1.0 + 0.20 * s * X2,      0.20 * s * X1, 0.0],
        [0.20 * (1.0 - c) * X1,    1.0,            0.0],
        [0.0,                       0.0,            1.0]
    ])
    return F


def compute_all(X: np.ndarray, t2: float,
                N1: np.ndarray, N2: np.ndarray,
                reference_values: dict = None,
                label_prefix: str = ""):
    """
    Verilen değerlerle tüm gerekli büyüklükleri hesaplar:
        X   – referans (malzeme) koordinatları
        t2  – değerlendirme zamanı
        N1, N2 – referans konfigürasyondaki birim doğrultu vektörleri

    reference_values sözlüğü verilmişse, her büyüklük saklı referansa
    karşı karşılaştırılır ve ✓ / ✗ ile etiketlenir.
    """
    rv  = reference_values or {}
    pfx = label_prefix

    def ref(key):
        return rv.get(key, None)

    I = np.eye(3)

    # ── 1. Referans konum (zaten biliniyor; sadece görüntüle) ──────────────────
    section(f"{pfx}1. Reference configuration position X")
    check("X", X, ref("X"))

    # ── 2. Deformasyon gradyanı F ────────────────────────────────────────────
    section(f"{pfx}2. Deformation gradient F")
    F = deformation_gradient(X, t2)
    check("F", F, ref("F"))

    # ── 3. F'nin tersi ──────────────────────────────────────────────────────
    section(f"{pfx}3. Inverse deformation gradient F⁻¹")
    F_inv = inv(F)
    check("F_inv", F_inv, ref("F_inv"))

    # ── 4. Green (sağ Cauchy-Green) deformasyon tansörü C = FᵀF ────────────
    section(f"{pfx}4. Green deformation tensor C = FᵀF (right Cauchy-Green)")
    C = F.T @ F
    check("C", C, ref("C"))

    # ── 5. Cauchy deformation tensor c = F⁻ᵀ F⁻¹ ───────────────────────────────
    #
    # DOĞRU TANIM:  c = F⁻ᵀ F⁻¹
    #
    # CHATGPT HATASI (adım 5):
    #   ChatGPT formül olarak  c = F⁻ᵀ F⁻¹  yazmış,
    #   ancak baskıladığı matris  c⁻¹ = F Fᵀ  (Finger tansörü) ile aynı.
    #   Yani adım 5 ve adım 7'nin SAYISAL DEĞERLERİ birbirine karışmış;
    #   iki tansörün rolleri tamamen yer değiştirmiş.
    #
    section(f"{pfx}5. Cauchy deformation tensor  c = F⁻ᵀ F⁻¹  ← ChatGPT HATALI (c⁻¹=FFᵀ vermiş)")
    c = F_inv.T @ F_inv   # doğru hesap: F⁻ᵀ F⁻¹
    check("c", c, ref("c"))

    # ── 6. Piola deformasyon tansörü C⁻¹ ─────────────────────────────────────
    section(f"{pfx}6. Piola deformation tensor C⁻¹")
    C_inv = inv(C)
    check("C_inv", C_inv, ref("C_inv"))

    # ── 7. Finger tansörü  c⁻¹ = F Fᵀ ──────────────────────────────────────────
    #
    # DOĞRU TANIM:  c⁻¹ = F Fᵀ
    #
    # CHATGPT HATASI (adım 7):
    #   ChatGPT bu adımı "c⁻¹" sembolüyle doğru etiketlemiş,
    #   ancak baskıladığı matris  F⁻ᵀ F⁻¹  (Cauchy tansörü) ile aynı.
    #   Adım 5 ile 7 arasındaki sayısal değerler tam tersine çevrilmiş durumda.
    #
    section(f"{pfx}7. Finger tansörü  c⁻¹ = FFᵀ  ← ChatGPT HATALI (c=F⁻ᵀF⁻¹ vermiş)")
    c_inv = F @ F.T   # doğru hesap: c⁻¹ = F Fᵀ
    check("c⁻¹ (Finger)", c_inv, ref("c_inv"))

    # ── 8. Lagrange genleme tansörü E = ½(C - I) ────────────────────────────
    section(f"{pfx}8. Lagrangian (Green-Lagrange) strain tensor E = ½(C - I)")
    E = 0.5 * (C - I)
    check("E", E, ref("E"))

    # ── 9. Eulerian (Almansi) strain tensor  e = ½(I - c) ───────────────────────
    #
    # DOĞRU TANIM:  e = ½(I - c)   burada  c = F⁻ᵀ F⁻¹
    #
    # CHATGPT HATASI (adım 9):
    #   ChatGPT adım 5'te c yerine yanlışlıkla c⁻¹ = FFᵀ kullandığından,
    #   bu adımdaki hesap fiilen  e = ½(I - c⁻¹)  haline gelmiş.
    #   Bu, Euler genleme tansörünün tamamen yanlış sonuç vermesine yol açıyor.
    #   Hata zinciri:  adım 5 yanlış  →  adım 9 yanlış.
    #
    section(f"{pfx}9. Euler (Almansi) strain  e = ½(I − c)  ← ChatGPT HATALI (c yerine c⁻¹ kullanmış)")
    e = 0.5 * (I - c)   # doğru hesap: ½(I - F⁻ᵀF⁻¹)
    check("e", e, ref("e"))

    # ── 10. Yer değiştirme U = x(t2) - X ──────────────────────────────────────
    section(f"{pfx}10. Displacement vector U = x(t₂) - X")
    s2, c2 = np.sin(t2), np.cos(t2)
    x_t2 = np.array([
        X[0] + 0.20 * s2 * X[0] * X[1],
        X[1] + 0.10 * (1 - c2) * X[0]**2,
        X[2]
    ])
    U = x_t2 - X
    check("U", U, ref("U"))

    # ── 11. Yer değiştirme gradyanı ∇U ile Lagrange E ────────────────────────
    section(f"{pfx}11. E via displacement gradient  E = ½(∇U + ∇Uᵀ + ∇Uᵀ∇U)")
    nabla_U = F - I   # yer değiştirme gradyanı H = F - I
    E_U = 0.5 * (nabla_U + nabla_U.T + nabla_U.T @ nabla_U)
    print(f"Displacement gradient ∇U:\n{nabla_U}\n")
    check("E (from U)", E_U, ref("E"))   # adım 8 ile eşleşmeli

    # ── 12. Doğrusallaştırılmış (sonsuz küçük) genleme tansörü Ḽ = ½(∇U + ∇Uᵀ) ────
    section(f"{pfx}12. Linearised strain tensor Ẽ = ½(∇U + ∇Uᵀ)  [nonlinear term dropped]")
    E_lin = 0.5 * (nabla_U + nabla_U.T)
    check("E_linear", E_lin, ref("E_lin"))

    # ── 13 & 14. Gerilme oranı Λ ve uzama E_N = Λ - 1 ─────────────────
    section(f"{pfx}13–14. Stretch ratio Λ and elongation E_N = Λ - 1")
    Lambda_N1 = float(np.sqrt(N1 @ C @ N1))
    Lambda_N2 = float(np.sqrt(N2 @ C @ N2))
    E_N1 = Lambda_N1 - 1.0
    E_N2 = Lambda_N2 - 1.0

    check("Lambda_N1", Lambda_N1, ref("Lambda_N1"))
    check("Lambda_N2", Lambda_N2, ref("Lambda_N2"))
    check("E_N1 (elongation)", E_N1, ref("E_N1"))
    check("E_N2 (elongation)", E_N2, ref("E_N2"))

    # ── 15. Deformasyon öncesi açı, deformasyon sonrası açı, değişim ─────────────
    section(f"{pfx}15. Angle change between N1 and N2")

    # Başlangıç açısı
    cos_theta0 = float(np.clip(N1 @ N2, -1, 1))
    theta0 = float(np.degrees(np.arccos(cos_theta0)))

    # Deforme doğrultular
    n1 = (F @ N1) / Lambda_N1
    n2 = (F @ N2) / Lambda_N2
    cos_theta = float(np.clip(n1 @ n2, -1, 1))
    theta  = float(np.degrees(np.arccos(cos_theta)))
    d_theta = theta - theta0

    check("theta0 (degrees)",  theta0,  ref("theta0"))
    check("theta  (degrees)",  theta,   ref("theta"))
    check("Delta_theta (deg)", d_theta, ref("d_theta"))
    print(f"  n1 (deformed N1/Λ₁) = {n1}")
    print(f"  n2 (deformed N2/Λ₂) = {n2}\n")

    # Kısım II'nin sonuçları yeniden kullanabilmesi için özet sözlük döndür
    return dict(F=F, F_inv=F_inv, C=C, c=c, C_inv=C_inv, c_inv=c_inv,
                E=E, e=e, U=U, E_lin=E_lin,
                Lambda_N1=Lambda_N1, Lambda_N2=Lambda_N2,
                E_N1=E_N1, E_N2=E_N2,
                theta0=theta0, theta=theta, d_theta=d_theta,
                n1=n1, n2=n2)


# ─────────────────────────────────────────────────────────────────────────────
# ChatGPT çözümündeki (PDF) referans değerler
# Kısım I doğrulaması için temel alınan değerler
# ─────────────────────────────────────────────────────────────────────────────

CHATGPT_REF = {
    # 1. Referans konum
    "X": np.array([1.0, 0.44, 0.0]),

    # 2. Deformasyon gradyanı
    "F": np.array([
        [1.06222540, 0.14142136, 0.0],
        [0.05857864, 1.00000000, 0.0],
        [0.0,        0.0,        1.0]
    ]),

    # 3. F'nin tersi
    "F_inv": np.array([
        [ 0.94881960, -0.13418336, 0.0],
        [-0.05558057,  1.00786028, 0.0],
        [ 0.0,         0.0,        1.0]
    ]),

    # 4. Green (sağ Cauchy-Green) C = FᵀF
    "C": np.array([
        [1.13175425, 0.20880000, 0.0],
        [0.20880000, 1.02000000, 0.0],
        [0.0,        0.0,        1.0]
    ]),

    # 5. ChatGPT'nin adım 5'te yazdığı YANLIŞ Cauchy değerleri (PDF'deki rakamlar):
    #    ChatGPT, Cauchy tansörü için  c = F⁻ᵀ F⁻¹  formülünü yazmış,
    #    ancak baskıladığı sayılar  c⁻¹ = F Fᵀ  (Finger tansörü) ile örtüşüyor.
    #    Yani 5. ve 7. adımlardaki SAYISAL DEĞERLER birbiriyle yer değiştirmiş.
    #    Bu referans olarak ChatGPT'nin PDF'deki hatalı matris saklanıyor;
    #    kod F⁻ᵀ F⁻¹ ile doğru değeri hesaplayacak ve ✗ WRONG basacak.
    "c": np.array([
        [1.14832279, 0.20364508, 0.0],   # ChatGPT'nin hatalı "c" değeri  (aslında c⁻¹ = FFᵀ)
        [0.20364508, 1.00343146, 0.0],
        [0.0,        0.0,        1.0]
    ]),

    # 6. Piola C⁻¹
    "C_inv": np.array([
        [ 0.91826381, -0.18797400, 0.0],
        [-0.18797400,  1.01887154, 0.0],
        [ 0.0,         0.0,        1.0]
    ]),

    # 7. ChatGPT'nin adım 7'de yazdığı YANLIŞ Finger değerleri (PDF'deki rakamlar):
    #    ChatGPT bu adımı  c⁻¹ = F Fᵀ  olarak doğru tanımlamış,
    #    fakat sayısal değer olarak  c = F⁻ᵀ F⁻¹  sonucunu (Cauchy'yi) koymuş.
    #    5. ve 7. adımların sayıları tam tersine çevrilmiş durumda.
    #    Bu referans ChatGPT'nin hatalı c⁻¹ matrisini saklıyor;
    #    kod F Fᵀ ile doğru değeri hesaplayacak ve ✗ WRONG basacak.
    "c_inv": np.array([
        [0.90334784, -0.18333324, 0.0],   # ChatGPT'nin hatalı "c⁻¹" değeri (aslında c = F⁻ᵀF⁻¹)
        [-0.18333324, 1.03378751, 0.0],
        [0.0,         0.0,        1.0]
    ]),

    # 8. Lagrange genleme E = ½(C-I)
    "E": np.array([
        [0.06587713, 0.10440000, 0.0],
        [0.10440000, 0.01000000, 0.0],
        [0.0,        0.0,        0.0]
    ]),

    # 9. ChatGPT'nin adım 9'da yazdığı YANLIŞ Euler genleme değerleri (PDF'deki rakamlar):
    #    Euler (Almansi) genleme tansörü  e = ½(I - c)  olarak tanımlanır.
    #    ChatGPT adım 5'te c yerine yanlışlıkla c⁻¹ = FFᵀ kullandığından,
    #    buradaki e hesabı da  e = ½(I - c⁻¹)  ile yapılmış → tamamen yanlış.
    #    Doğru hesap  e = ½(I - F⁻ᵀF⁻¹)  ile yapılmalıdır.
    "e": np.array([
        [-0.07416139, -0.10182254, 0.0],   # ChatGPT'nin hatalı e değeri (c yerine c⁻¹ kullanılmış)
        [-0.10182254, -0.00171573, 0.0],
        [ 0.0,         0.0,        0.0]
    ]),

    # 10. Yer değiştirme
    "U": np.array([0.06222540, 0.02928932, 0.0]),

    # 12. Doğrusallaştırılmış genleme
    "E_lin": np.array([
        [0.06222540, 0.10000000, 0.0],
        [0.10000000, 0.00000000, 0.0],
        [0.0,        0.0,        0.0]
    ]),

    # 13–14. Gerilme oranı ve uzama
    "Lambda_N1": 1.06383939,
    "Lambda_N2": 1.13343598,
    "E_N1":      0.06383939,
    "E_N2":      0.13343598,

    # 15. Açılar
    "theta0":  45.0,
    "theta":   38.17435996,
    "d_theta": -6.82564004,
}


# ─────────────────────────────────────────────────────────────────────────────
# KISIM I – ChatGPT çözümünü doğrula
# ─────────────────────────────────────────────────────────────────────────────

def run_part_I():
    header("PART I – VERIFICATION OF CHATGPT SOLUTION")

    print("""
  Motion:  x1 = X1 + 0.20·sin(t)·X1·X2
           x2 = X2 + 0.10·(1−cos(t))·X1²
           x3 = X3

  ChatGPT choices:
      t1 = π/2   → particle observed at x = (1.088, 0.54, 0)
      t2 = π/4   → evaluate deformation tensors at this time
      N1 = [1, 0, 0],   N2 = [1/√2, 1/√2, 0]

  NOTE — ChatGPT'nin tespiti edilen hataları:

  Adım 5 (Cauchy  c = F⁻ᵀF⁻¹):
    ChatGPT formülü doğru yazmış, ancak baskıladığı sayı  c⁻¹ = FFᵀ  ile aynı.

  Adım 7 (Finger  c⁻¹ = FFᵀ):
    ChatGPT "c⁻¹" etiketiyle doğru kavramı kastetmiş, ancak baskıladığı sayı
    F⁻ᵀF⁻¹ (Cauchy) ile aynı.  → Adım 5 ve 7'nin sayısal değerleri yer değiştirmiş.

  Adım 9 (Euler genleme  e = ½(I - c)):
    Cauchy (c) yanlış olduğundan bu hesap da yanlış.
    Doğru formül  e = ½(I - F⁻ᵀF⁻¹),  ChatGPT ise  ½(I - c⁻¹)  hesaplamış.
""")

    # 1. t1 = π/2 anında x = (1.088, 0.54, 0) konumundan referans konum bul
    t1  = np.pi / 2
    t2  = np.pi / 4
    x_obs = np.array([1.088, 0.54, 0.0])

    X = find_reference_position(x_obs, t1)

    N1 = np.array([1.0, 0.0, 0.0])
    N2 = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2), 0.0])

    compute_all(X, t2, N1, N2, reference_values=CHATGPT_REF, label_prefix="[Part I] ")


# ─────────────────────────────────────────────────────────────────────────────
# KISIM II – Kendi seçimlerimiz
# ─────────────────────────────────────────────────────────────────────────────

def run_part_II():
    header("PART II – OUR OWN CHOICES")

    print("""
  ── Seçimler ────────────────────────────────────────────────────────────────
  t1 = π/3

  Referans parçacık: X = (0.8, 0.3, 0)  →  [0,1]² içinde yer alıyor.
  t1 anında ileri hareket:  x = (0.842, 0.332, 0)
  X1 için kubiğin kökleri ≈ {0.8, +10.6, −11.4} olduğundan
  [0,1]² içinde tek anlamlı kök var  → tek çözüm ✓

  t2 = π/5  (36°)

  Doğrultular:
      N1 = [1, 0, 0]              (X1 ekseni)
      N2 = [0.5, √3/2, 0]        (N1 ile 60° açı yapıyor → θ₀ = 60°)
  ─────────────────────────────────────────────────────────────────────────────
""")

    # ── KISIM II SEÇİMLERİ ───────────────────────────────────────────────────
    # t1: parçacığın gözlemlendiği an.
    t1_MY = np.pi / 3                              # t1 = π/3  ≈ 1.047 rad (60°)

    # t1 anında parçacığın uzaysal konumu.
    # X=(0.8, 0.3, 0) noktasının ileri hareketi:
    #   x1 = 0.8 + 0.20·sin(π/3)·0.8·0.3 = 0.84156...  → 0.842 olarak alındı
    #   x2 = 0.3 + 0.10·(1−cos(π/3))·0.8² = 0.33200     (tam değer)
    x_obs_MY = np.array([0.842, 0.332, 0.0])

    # t2: şekil değiştirme tansörlerinin hesaplandığı an.
    # t1'den farklı, katı olmayan bir değer.
    t2_MY = np.pi / 5                              # t2 = π/5  ≈ 0.628 rad (36°)

    # Referans konfigürasyondaki doğrultu vektörleri (birim vektör olmalı).
    # N1 ile N2 arasındaki başlangıç açısı 60°.
    N1_MY = np.array([1.0, 0.0, 0.0])             # N1 = e1
    N2_MY = np.array([0.5, np.sqrt(3)/2, 0.0])    # N2 = [cos60°, sin60°, 0]
    # ── SEÇİMLER SONU ───────────────────────────────────────────────────────

    # Doğrultu vektörlerini normalize et (güvenlik)
    N1_MY = N1_MY / norm(N1_MY)
    N2_MY = N2_MY / norm(N2_MY)

    print(f"  t1     = {t1_MY:.6f} rad  ({np.degrees(t1_MY):.4f}°)")
    print(f"  x_obs  = {x_obs_MY}")
    print(f"  t2     = {t2_MY:.6f} rad  ({np.degrees(t2_MY):.4f}°)")
    print(f"  N1     = {N1_MY}")
    print(f"  N2     = {N2_MY}\n")

    # Referans konumu çöz
    X_MY = find_reference_position(x_obs_MY, t1_MY)

    # Bulunan X'in t1 anında x_obs'a geri dönüştüğünü doğrula
    s1, c1 = np.sin(t1_MY), np.cos(t1_MY)
    x1_back = X_MY[0] + 0.20 * s1 * X_MY[0] * X_MY[1]
    x2_back = X_MY[1] + 0.10 * (1 - c1) * X_MY[0]**2
    x_back  = np.array([x1_back, x2_back, X_MY[2]])
    residual = norm(x_back - x_obs_MY)

    print(f"  Residual (should be ~0): {residual:.2e}")
    if residual > 1e-6:
        print(f"  {RED}WARNING: Large residual — check your x_obs or t1 choice!{RESET}\n")
    else:
        print(f"  {GREEN}✓ Forward mapping verified.{RESET}\n")

    # Tüm büyüklükleri hesapla (dış referans yok → ✓/✗ etiketi yok,
    # ancak tüm büyüklükler açıkça yazdırılır)
    compute_all(X_MY, t2_MY, N1_MY, N2_MY,
                reference_values=None, label_prefix="[Part II] ")


# ─────────────────────────────────────────────────────────────────────────────
# Ana giriş noktası
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_part_I()
    run_part_II()

    header("DONE")
    print("  ✓ = value matches the ChatGPT reference (within tol=1e-5)")
    print("  ✗ = value differs from the ChatGPT reference\n")
    print("  For Part II no reference is available so results are printed")
    print("  without ✓/✗ labels — interpret and compare manually.\n")
