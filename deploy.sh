#!/bin/bash
# Script de deploy contínuo automático
set -e

MESSAGE=${1:-"auto-deploy: updates and improvements"}

echo "🚀 Iniciando deploy automático..."
git add .
if git diff-index --quiet HEAD --; then
    echo "Nenhuma alteração para comitar."
else
    git commit -m "$MESSAGE"
    git push origin main
    echo "✓ Alterações enviadas para o GitHub (danilomenezes88/prevision)!"
    echo "✓ O Streamlit Cloud já detectou e iniciou o redeploy automático."
fi
