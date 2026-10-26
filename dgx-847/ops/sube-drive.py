#!/usr/bin/env python3
"""Sube un fichero a internal-domain y escribe por stdout una URL prefirmada del endpoint INTERNO, que es la que alcanza el
whatsapp-connector (memoria gotcha_whatsapp_adjuntos_presigned_drive). Credenciales del secret internal-domain
(ns whatsapp-mcp), sin imprimirlas; la subida va por un port-forward al servicio.
  python3 sube-drive.py FICHERO CLAVE [Content-Type]"""
import base64, json, os, socket, subprocess, sys, time
import boto3
from botocore.config import Config

fich, clave = sys.argv[1], sys.argv[2]
ctype = sys.argv[3] if len(sys.argv) > 3 else "text/html; charset=utf-8"
env = dict(os.environ, KUBECONFIG=os.path.expanduser("~/.kube/config"))
s = json.loads(subprocess.run(["kubectl", "get", "secret", "-n", "whatsapp-mcp", "internal-domain", "-o", "jsonpath={.data}"],
                              capture_output=True, text=True, env=env, check=True).stdout)
s = {k: base64.b64decode(v).decode() for k, v in s.items()}
with socket.socket() as so:
    so.bind(("127.0.0.1", 0)); port = so.getsockname()[1]
pf = subprocess.Popen(["kubectl", "port-forward", "-n", "backup-hub", "svc/internal-domain", f"{port}:9000"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
try:
    time.sleep(3)
    kw = dict(aws_access_key_id=s["AWS_ACCESS_KEY_ID"], aws_secret_access_key=s["AWS_SECRET_ACCESS_KEY"],
              region_name=s.get("S3_REGION") or s.get("AWS_REGION") or "us-east-1", config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"))
    boto3.client("s3", endpoint_url=f"http://127.0.0.1:{port}", **kw).upload_file(fich, s["S3_BUCKET"], clave, ExtraArgs={"ContentType": ctype})
    print(boto3.client("s3", endpoint_url=s["S3_ENDPOINT"], **kw).generate_presigned_url(
        "get_object", Params={"Bucket": s["S3_BUCKET"], "Key": clave}, ExpiresIn=7 * 86400))
finally:
    pf.terminate()
