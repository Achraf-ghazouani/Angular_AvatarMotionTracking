#!/bin/bash

# Script d'installation automatique pour Angular Avatar Motion Tracking
# Usage: ./install.sh

echo "🎭 Avatar IA - Motion Tracking"
echo "================================"
echo ""

# Vérifier Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js n'est pas installé"
    echo "📥 Téléchargez Node.js depuis: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node -v)
echo "✅ Node.js détecté: $NODE_VERSION"

# Vérifier npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm n'est pas installé"
    exit 1
fi

NPM_VERSION=$(npm -v)
echo "✅ npm détecté: $NPM_VERSION"

echo ""
echo "📦 Installation des dépendances..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Erreur lors de l'installation"
    echo "💡 Essayez: npm install --legacy-peer-deps"
    exit 1
fi

echo ""
echo "✅ Installation réussie!"
echo ""
echo "🚀 Pour démarrer l'application:"
echo "   npm start"
echo ""
echo "📖 Documentation complète: README.md"
echo "🆘 En cas de problème: TROUBLESHOOTING.md"
echo ""
echo "Bon développement! 🎉"
