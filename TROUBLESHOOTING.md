# 🔧 Guide de Dépannage

Ce guide vous aide à résoudre les problèmes courants.

## 📋 Table des Matières

- [Installation](#installation)
- [Webcam](#webcam)
- [Performance](#performance)
- [Tracking](#tracking)
- [Avatar](#avatar)
- [Build & Déploiement](#build--déploiement)

---

## Installation

### ❌ Erreur : "npm install" échoue

**Symptômes :**
```
npm ERR! code ERESOLVE
npm ERR! ERESOLVE unable to resolve dependency tree
```

**Solutions :**

1. **Nettoyer le cache npm**
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

2. **Utiliser la bonne version de Node.js**
```bash
node --version  # Devrait être >= 18.0.0
```

3. **Utiliser --legacy-peer-deps**
```bash
npm install --legacy-peer-deps
```

### ❌ Erreur : "No matching version found for kalidokit"

**Symptômes :**
```
npm error notarget No matching version found for kalidokit@^1.1.6
```

**Solution :**
La version 1.1.6 de kalidokit n'existe pas. La version maximale disponible est 1.1.5. Vérifiez que votre `package.json` utilise la bonne version :
```json
"kalidokit": "^1.1.5"
```

### ❌ Erreur : TypeScript compilation failed

**Symptômes :**
```
Error: node_modules/@angular/core/index.d.ts is missing
```

**Solution :**
```bash
npm install --save-dev @angular/core @angular/common
npm install
```

---

## Webcam

### ❌ La webcam ne s'active pas

**Symptômes :**
- Le bouton "Start Tracking" ne fait rien
- Message d'erreur : "Failed to access webcam"

**Solutions :**

1. **Vérifier les permissions du navigateur**

**Chrome/Edge :**
```
1. Cliquer sur l'icône 🔒 dans la barre d'adresse
2. Permissions > Caméra > Autoriser
3. Actualiser la page
```

**Firefox :**
```
1. about:preferences#privacy
2. Permissions > Caméra
3. Autoriser l'URL localhost
```

2. **Vérifier que la webcam fonctionne**
```
Windows: Ouvrir "Caméra"
macOS: Ouvrir "Photo Booth"
Linux: cheese ou guvcview
```

3. **Vérifier qu'aucune autre application n'utilise la webcam**
```
Fermer: Zoom, Teams, Skype, OBS, etc.
```

4. **Tester avec HTTPS**
```bash
# Générer un certificat SSL local
npm install -g mkcert
mkcert -install
mkcert localhost

# Modifier angular.json pour ajouter SSL
```

### ❌ L'image de la webcam est floue

**Solutions :**
1. Nettoyer la lentille de la webcam
2. Améliorer l'éclairage de la pièce
3. Augmenter la résolution dans le code :

```typescript
// src/app/services/tracking.service.ts
video: {
  width: { ideal: 1920 },  // Au lieu de 1280
  height: { ideal: 1080 }, // Au lieu de 720
  frameRate: { ideal: 30 }
}
```

---

## Performance

### ❌ FPS trop faible (< 24)

**Symptômes :**
- L'animation est saccadée
- Le compteur FPS est rouge
- Latence > 150ms

**Solutions :**

1. **Réduire la complexité du modèle MediaPipe**
```typescript
// Dans Settings ou src/app/models/config.model.ts
mediapipe: {
  modelComplexity: 0,  // 0 = Rapide, 1 = Équilibré, 2 = Précis
  smoothLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
}
```

2. **Désactiver les features inutiles**
```typescript
mediapipe: {
  enableSegmentation: false,  // Désactiver la segmentation
  smoothSegmentation: false
}
```

3. **Réduire la résolution de la webcam**
```typescript
video: {
  width: { ideal: 640 },   // Réduire à 640x480
  height: { ideal: 480 },
  frameRate: { ideal: 30 }
}
```

4. **Fermer les autres applications**
- Fermer les onglets de navigateur inutiles
- Fermer les applications lourdes (Photoshop, etc.)
- Vérifier l'utilisation CPU/GPU dans le Gestionnaire des tâches

5. **Vérifier le GPU**
```
Chrome: chrome://gpu
Vérifier que WebGL est activé et hardware accelerated
```

6. **Optimiser Three.js**
```typescript
threejs: {
  antialias: false,  // Désactiver l'antialiasing
  powerPreference: 'high-performance'
}
```

### ❌ Latence élevée (> 100ms)

**Solutions :**

1. **Réduire le buffer de smoothing**
```typescript
// src/app/services/tracking.service.ts
private readonly SMOOTHING_WINDOW = 3;  // Au lieu de 5
```

2. **Désactiver l'IA temporairement**
```typescript
ai: {
  enabled: false
}
```

### ❌ L'application consomme trop de mémoire

**Symptômes :**
- Le navigateur ralentit avec le temps
- Message "Out of memory"

**Solutions :**

1. **Nettoyer les buffers régulièrement**
```typescript
// Ajouter dans le service de tracking
private cleanupBuffers(): void {
  if (this.smoothingBuffer.length > 100) {
    this.smoothingBuffer = this.smoothingBuffer.slice(-50);
  }
}
```

2. **Redémarrer l'application périodiquement**
```
Recharger la page toutes les heures
```

---

## Tracking

### ❌ L'avatar ne bouge pas

**Symptômes :**
- La webcam fonctionne
- Le tracking est actif (vert)
- Mais l'avatar reste immobile

**Solutions :**

1. **Vérifier la console**
```
F12 > Console
Chercher des erreurs rouges
```

2. **Vérifier l'éclairage**
```
- Éviter les contre-jours
- Avoir une lumière frontale
- Éviter les ombres sur le visage
```

3. **Se positionner correctement**
```
- Visage entièrement visible
- Distance 50-100cm de la webcam
- Fond neutre si possible
```

4. **Vérifier que Kalidokit fonctionne**
```typescript
// Dans la console du navigateur
console.log(window.Kalidokit);  // Devrait afficher l'objet Kalidokit
```

### ❌ Les mouvements sont inversés

**Solution :**
```typescript
// Ajouter dans animation.service.ts
if (results.Face?.head?.degrees) {
  const { x, y, z } = results.Face.head.degrees;
  headBone.rotation.set(
    THREE.MathUtils.degToRad(x),
    THREE.MathUtils.degToRad(-y),  // Inverser Y
    THREE.MathUtils.degToRad(z)
  );
}
```

### ❌ Les mouvements sont trop sensibles/insensibles

**Solutions :**

1. **Ajuster le smoothing**
```typescript
private readonly SMOOTHING_WINDOW = 7;  // Plus = plus lisse
```

2. **Ajuster les facteurs de rotation**
```typescript
headBone.rotation.set(
  THREE.MathUtils.degToRad(x * 0.5),  // Réduire la sensibilité
  THREE.MathUtils.degToRad(y * 0.5),
  THREE.MathUtils.degToRad(z * 0.5)
);
```

### ❌ Le tracking perd le visage

**Solutions :**
1. Augmenter la confiance de détection
```typescript
mediapipe: {
  minDetectionConfidence: 0.7,  // Au lieu de 0.5
  minTrackingConfidence: 0.7
}
```

2. Améliorer l'éclairage
3. Éviter les mouvements brusques

---

## Avatar

### ❌ L'avatar ne se charge pas

**Symptômes :**
- Message "Avatar loaded" n'apparaît pas
- Erreur dans la console

**Solutions :**

1. **Vérifier le chemin du fichier**
```typescript
// src/app/models/config.model.ts
avatar: {
  modelPath: 'assets/models/avatar.glb',  // Vérifier le chemin
}
```

2. **Vérifier que le fichier existe**
```bash
ls -la src/assets/models/
# Devrait afficher avatar.glb
```

3. **Vérifier le format**
```
Formats supportés: GLB, GLTF
Formats non supportés: FBX (nécessite FBXLoader)
```

4. **Utiliser l'avatar de secours**
```
Si aucun modèle n'est trouvé, un avatar simple sera créé automatiquement
```

### ❌ L'avatar est trop grand/petit

**Solution :**
```typescript
avatar: {
  scale: 0.5,  // Réduire l'échelle
  position: { x: 0, y: -2, z: 0 }  // Ajuster la position
}
```

### ❌ L'avatar est mal orienté

**Solution :**
```typescript
avatar: {
  rotation: { 
    x: 0, 
    y: Math.PI,  // Rotation de 180°
    z: 0 
  }
}
```

---

## Build & Déploiement

### ❌ Le build échoue

**Symptômes :**
```
npm run build
ERROR in ...
```

**Solutions :**

1. **Nettoyer et rebuild**
```bash
rm -rf .angular dist
npm run build
```

2. **Vérifier les erreurs TypeScript**
```bash
npx tsc --noEmit
```

3. **Augmenter la mémoire Node.js**
```bash
# Windows
set NODE_OPTIONS=--max_old_space_size=4096
npm run build

# macOS/Linux
NODE_OPTIONS=--max_old_space_size=4096 npm run build
```

### ❌ Le bundle est trop gros (> 20MB)

**Solutions :**

1. **Vérifier la taille**
```bash
npm run build
ls -lh dist/avatar-motion-tracking/browser/
```

2. **Analyser le bundle**
```bash
npm install -g webpack-bundle-analyzer
npm run build -- --stats-json
npx webpack-bundle-analyzer dist/avatar-motion-tracking/stats.json
```

3. **Lazy load des modules**
```typescript
// Charger ONNX seulement si nécessaire
if (config.ai.enabled) {
  const ort = await import('onnxruntime-web');
}
```

### ❌ La webcam ne fonctionne pas en production

**Cause :** HTTPS requis

**Solutions :**

1. **Activer HTTPS sur votre serveur**
```nginx
server {
  listen 443 ssl;
  ssl_certificate /path/to/cert.pem;
  ssl_certificate_key /path/to/key.pem;
}
```

2. **Utiliser un service avec HTTPS**
- Netlify (HTTPS automatique)
- Vercel (HTTPS automatique)
- GitHub Pages (HTTPS automatique)

---

## 🆘 Support Avancé

### Logs de Debug

Activer les logs détaillés :
```typescript
// Dans app.component.ts
ngOnInit() {
  console.log('🐛 DEBUG MODE ENABLED');
  // Logs détaillés...
}
```

### Informations Système

```typescript
console.log('System Info:', {
  userAgent: navigator.userAgent,
  platform: navigator.platform,
  webgl: (() => {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    return gl ? 'supported' : 'not supported';
  })(),
  mediaDevices: 'mediaDevices' in navigator
});
```

### Tester MediaPipe Isolé

```html
<!-- test-mediapipe.html -->
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/holistic"></script>
</head>
<body>
  <video id="video" width="640" height="480" autoplay></video>
  <script>
    // Test basique de MediaPipe
  </script>
</body>
</html>
```

---

## 📞 Obtenir de l'Aide

Si vous ne trouvez pas de solution :

1. **Chercher dans les Issues GitHub**
   - [Issues existantes](https://github.com/Achraf-ghazouani/Angular_AvatarMotionTracking/issues)

2. **Créer une nouvelle Issue**
   - Utiliser le template de Bug Report
   - Inclure les logs de console
   - Préciser l'environnement

3. **Communauté**
   - Discussions GitHub (à venir)
   - Stack Overflow (tag: `angular avatar-tracking`)

---

**N'oubliez pas de consulter la [documentation complète](README.md) ! 📚**
