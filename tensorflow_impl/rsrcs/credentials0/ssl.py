from OpenSSL import crypto
from datetime import datetime

cert_file = input()
cert = crypto.load_certificate(crypto.FILETYPE_PEM, open(cert_file).read())

subject = cert.get_subject()
issued_to = subject.CN    
issuer = cert.get_issuer()
issued_by = issuer.CN

print("Common Name: " , issued_to)
print("Valid From: " , datetime.strptime(cert.get_notAfter().decode('utf-8'),'%Y%m%d%H%M%SZ'))
print("Valid To: " , datetime.strptime(cert.get_notBefore().decode('utf-8'),'%Y%m%d%H%M%SZ'))
print("Issuer: " , issued_by) 