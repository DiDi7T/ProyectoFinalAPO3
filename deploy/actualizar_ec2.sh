#!/usr/bin/env bash
# Sube el codigo actual (y los modelos de experiments/checkpoints/, si existen)
# a la instancia EC2 y reinicia la interfaz.
#
#   bash deploy/actualizar_ec2.sh
#
# Variables opcionales: HOST (ip publica) y LLAVE (ruta de la llave SSH).
set -euo pipefail

HOST="${HOST:-100.52.186.15}"
LLAVE="${LLAVE:-$HOME/.ssh/apo3-frutas.pem}"
DESTINO="ubuntu@$HOST"

cd "$(dirname "$0")/.."
PAQUETE="$(mktemp -t app).tgz"

# Codigo versionado + archivos nuevos sin versionar, sin notebooks ni el PDF.
git ls-files -co --exclude-standard | grep -vE '\.(pdf|ipynb)$' \
  | COPYFILE_DISABLE=1 tar czf "$PAQUETE" -T -

# Modelos entrenados, si estan en local.
MODELOS=$(ls experiments/checkpoints/{cnn_mejor.pt,mejor_classic_quality.joblib,mejor_classic_size.joblib} 2>/dev/null || true)

scp -i "$LLAVE" "$PAQUETE" "$DESTINO:/tmp/app.tgz"
ssh -i "$LLAVE" "$DESTINO" 'tar xzf /tmp/app.tgz -C ~/app 2>/dev/null'
if [ -n "$MODELOS" ]; then
  echo "Subiendo modelos: $MODELOS"
  scp -i "$LLAVE" $MODELOS "$DESTINO:~/app/experiments/checkpoints/"
fi
# Dependencias de la interfaz (torch ya esta instalado en version CPU).
ssh -i "$LLAVE" "$DESTINO" 'cd ~/app && .venv/bin/pip install -q numpy pandas joblib \
  opencv-python-headless "Pillow>=11" pillow-heif "scikit-learn==1.9.0" "rembg[cpu]" streamlit \
  && .venv/bin/pip install -q torchvision --index-url https://download.pytorch.org/whl/cpu \
  && .venv/bin/pip uninstall -y -q opencv-python 2>/dev/null; \
  .venv/bin/pip install -q --force-reinstall --no-deps opencv-python-headless'
ssh -i "$LLAVE" "$DESTINO" 'sudo systemctl restart frutas && systemctl is-active frutas'
rm -f "$PAQUETE"

echo "Listo: https://${HOST//./-}.sslip.io"
