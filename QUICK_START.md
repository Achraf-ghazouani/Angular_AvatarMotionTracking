# 🚀 Guide de Démarrage Rapide

Ce guide vous permet de lancer l'application en 5 minutes.

## ⚡ Installation Express

### 1. Prérequis
- Node.js >= 18.0 ([Télécharger](https://nodejs.org/))
- Un navigateur moderne (Chrome, Edge, ou Firefox)
- Une webcam fonctionnelle

### 2. Installation

```bash
# Cloner le projet
git clone https://github.com/Achraf-ghazouani/Angular_AvatarMotionTracking.git
cd Angular_AvatarMotionTracking

# Installer les dépendances
npm install

# Lancer l'application
npm start
```

### 3. Accéder à l'Application

Ouvrir votre navigateur à l'adresse : **http://localhost:4200**

## 🎯 Première Utilisation

### Étape 1 : Autoriser la Webcam
- Le navigateur vous demandera l'autorisation d'accéder à votre webcam
- **Cliquez sur "Autoriser"**

### Étape 2 : Démarrer le Tracking
- Cliquez sur le bouton **"Start Tracking"** dans le panneau gauche
- Attendez que l'indicateur "Tracking" devienne vert

### Étape 3 : Positionner votre Visage
- Placez-vous face à la webcam
- Assurez-vous d'avoir un bon éclairage
- L'avatar devrait commencer à reproduire vos mouvements

### Étape 4 : Tester l'Interaction
- Essayez de cliquer sur le **cube rouge** dans la scène 3D
- Utilisez la souris pour naviguer :
  - **Clic gauche + glisser** : Rotation
  - **Molette** : Zoom
  - **Clic droit + glisser** : Déplacement

## 📊 Vérifier les Performances

Dans le panneau "Performance", vous devriez voir :
- **FPS** : ~30 (vert si bon)
- **Latency** : <100ms (vert si bon)
- **Quality** : >90% (vert si bon)

### Si les performances sont faibles :

1. **Ouvrir les Paramètres** (bouton Settings)
2. **Réduire "Model Complexity"** à "Low (Fast)"
3. Le tracking sera moins précis mais plus rapide

## ❌ Problèmes Courants

### La webcam ne fonctionne pas
```
Solution: Vérifier les permissions de votre navigateur
Chrome: chrome://settings/content/camera
Firefox: about:preferences#privacy
```

### L'application ne démarre pas
```bash
# Nettoyer et réinstaller
rm -rf node_modules package-lock.json
npm install
npm start
```

### FPS trop faible
```
1. Fermer les autres applications
2. Réduire la qualité dans Settings
3. Utiliser un navigateur basé sur Chromium (Chrome/Edge)
```

### L'avatar ne bouge pas
```
1. Vérifier que "Tracking" est actif (indicateur vert)
2. S'assurer d'être visible dans la webcam
3. Améliorer l'éclairage de la pièce
```

## 🎨 Ajouter un Avatar Personnalisé

### Option 1 : Ready Player Me (Recommandé)

1. Aller sur [Ready Player Me](https://readyplayer.me/)
2. Créer votre avatar
3. Télécharger en format **GLB**
4. Placer le fichier dans `src/assets/models/avatar.glb`
5. Redémarrer l'application

### Option 2 : Mixamo

1. Aller sur [Mixamo](https://www.mixamo.com/)
2. Choisir un personnage
3. Télécharger en format **FBX** ou **GLB**
4. Placer le fichier dans `src/assets/models/avatar.glb`
5. Redémarrer l'application

## 🤖 Activer l'IA (Avancé)

L'IA est **désactivée par défaut**. Pour l'activer :

1. **Entraîner un modèle** (voir [AI_TRAINING_GUIDE.md](AI_TRAINING_GUIDE.md))
2. **Exporter en ONNX** : `motion_correction.onnx`
3. **Placer le fichier** dans `src/assets/models/`
4. **Éditer la configuration** :

```typescript
// src/app/models/config.model.ts
ai: {
  enabled: true,  // Changer à true
  modelPath: 'assets/models/motion_correction.onnx',
  inferenceType: 'onnx'
}
```

5. **Redémarrer** l'application

## 🔧 Commandes Utiles

```bash
# Développement
npm start              # Lance le serveur de développement
npm run build          # Build pour la production
npm run watch          # Build avec watch mode

# Nettoyage
rm -rf .angular        # Nettoyer le cache Angular
rm -rf node_modules    # Supprimer les dépendances
npm install            # Réinstaller les dépendances
```

## 📱 Build de Production

```bash
# Créer un build optimisé
npm run build

# Les fichiers seront dans dist/avatar-motion-tracking/
# Servir avec un serveur statique
npx serve -s dist/avatar-motion-tracking
```

## 🌐 Déploiement

### Netlify / Vercel / GitHub Pages

```bash
# Build
npm run build

# Déployer le contenu de dist/avatar-motion-tracking/
```

⚠️ **Important** : 
- Nécessite **HTTPS** pour l'accès webcam
- Configurer les **redirections** pour le routing Angular

## 📚 Documentation Complète

Pour plus d'informations :
- [README.md](README.md) - Documentation complète
- [AI_TRAINING_GUIDE.md](AI_TRAINING_GUIDE.md) - Guide d'entraînement IA
- [CHANGELOG.md](CHANGELOG.md) - Historique des versions

## 💬 Support

Problème ? Questions ?
- [Issues GitHub](https://github.com/Achraf-ghazouani/Angular_AvatarMotionTracking/issues)
- Email: votre@email.com

## ✅ Checklist de Démarrage

- [ ] Node.js installé (v18+)
- [ ] Projet cloné
- [ ] Dépendances installées (`npm install`)
- [ ] Application lancée (`npm start`)
- [ ] Webcam autorisée
- [ ] Tracking démarré
- [ ] Avatar se déplace avec vos mouvements
- [ ] Performances acceptables (FPS > 24)

---

**Félicitations ! Vous êtes prêt à utiliser Avatar IA Motion Tracking ! 🎉**
