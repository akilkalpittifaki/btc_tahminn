from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization

# Özel Anahtar (Private Key) oluştur
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048,
)

# Özel Anahtarı PEM formatında dosyaya kaydet
with open("private_key.pem", "wb") as private_file:
    private_file.write(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

# Açık Anahtar (Public Key) oluştur
public_key = private_key.public_key()

# Açık Anahtarı PEM formatında dosyaya kaydet
with open("public_key.pem", "wb") as public_file:
    public_file.write(
        public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )

print("Public ve Private anahtarlar oluşturuldu ve kaydedildi.")
