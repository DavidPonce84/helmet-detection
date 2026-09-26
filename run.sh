#!/usr/bin/env bash
set -e

# Directorio base del proyecto
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

echo "=========================================================="
echo "🛡️  Iniciando Control de Cascos Nayón (Streamlit)..."
echo "=========================================================="

python3 -m streamlit run src/app.py --server.headless=false
