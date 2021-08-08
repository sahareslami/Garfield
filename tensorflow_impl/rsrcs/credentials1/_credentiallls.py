import os


def _load_credential_from_file(filepath):
    real_path = os.path.join(os.path.dirname(__file__), filepath)
    with open(real_path, 'rb') as f:
        return f.read()



def get_my_private_key(role):
    return _load_credential_from_file(role + ".key")

directory = r"./public_keys"
public_keys = {}

for filename in os.listdir(directory):
    if filename.endswith(".key"):
        public_keys[filename[:-4]] = _load_credential_from_file(os.path.join(directory, filename))
    else:
        continue


root_certificate = _load_credential_from_file("Server1.crt")
my_certificate = _load_credential_from_file("my_certificate.crt")


# SERVER_CERTIFICATE = _load_credential_from_file('credentials/localhost.crt')
# SERVER_CERTIFICATE_KEY = _load_credential_from_file('credentials/localhost.key')
# ROOT_CERTIFICATE = _load_credential_from_file('credentials/root.crt')
# CLIENT_CERTIFICATE = _load_credential_from_file('credentials/client.crt')
# CLIENT_CERTIFICATE_PRIVATE_KEY = _load_credential_from_file('credentials/client.key')
# CLIENT_CERTIFICATE_PUBLIC_KEY = _load_credential_from_file('credentials/client_pub.key')