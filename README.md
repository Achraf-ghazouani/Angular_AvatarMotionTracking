# 🎭 Avatar IA - Motion Tracking en Temps Réel

Système d'animation 3D interactif dans Angular + Three.js, capable de reproduire en temps réel les expressions et mouvements d'un utilisateur à partir d'une webcam, avec correction IA (PyTorch/ONNX).

**Supporte les avatars VRM, Mixamo (FBX/GLB) et GLB standard !**

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Angular](https://img.shields.io/badge/Angular-18+-red)
![Three.js](https://img.shields.io/badge/Three.js-0.160-green)
![VRM](https://img.shields.io/badge/VRM-✓-purple)
![Mixamo](https://img.shields.io/badge/Mixamo-✓-orange)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

## 📋 Table des Matières

- [Caractéristiques](#-caractéristiques)
- [Formats d'Avatars](#-formats-davatars-supportés)
- [Architecture](#-architecture)
- [Technologies](#-technologies)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Configuration](#️-configuration)
- [Phases de Développement](#-phases-de-développement)
- [Intégration IA](#-intégration-ia-phase-3)
- [Performance](#-performance)
- [Structure du Projet](#-structure-du-projet)
- [Contribution](#-contribution)

## ✨ Caractéristiques

### Phase 1 - Prototype ✅
- ✅ Tracking facial en temps réel avec **MediaPipe Holistic**
- ✅ Détection des expressions et mouvements du corps
- ✅ Animation 3D de l'avatar avec **Kalidokit**
- ✅ Rendu haute qualité avec **Three.js**
- ✅ **Support multi-format** : VRM, Mixamo, GLB

### Phase 2 - Stabilisation ✅
- ✅ Lissage des mouvements (moyenne mobile pondérée)
- ✅ Correction des erreurs de tracking Kalidokit
- ✅ Réduction du jitter et des mouvements brusques

### Phase 3 - IA 🚧
- 🚧 Infrastructure prête pour modèle LSTM PyTorch
- 🚧 Support ONNX.js pour l'inférence côté client
- 🚧 Prédiction et correction intelligente des mouvements

### Phase 4 - Interaction ✅
- ✅ Manipulation d'objets 3D avec Raycaster
- ✅ Cube interactif de test
- ✅ Système d'événements pour interactions futures

### Phase 5 - Sécurité ✅
- ✅ Traitement 100% côté client (frontend uniquement)
- ✅ Aucune donnée envoyée au serveur
- ✅ Confidentialité totale des données utilisateur

## 🎭 Formats d'Avatars Supportés

| Format | Extension | Source | Recommandé |
|--------|-----------|---------|------------|
| **VRM** | `.vrm` | VRoid Hub, VRoid Studio | ⭐⭐⭐ Meilleur |
| **Mixamo** | `.fbx`, `.glb` | Mixamo.com | ⭐⭐⭐ Idéal pour débuter |
| **Ready Player Me** | `.glb` | readyplayer.me | ⭐⭐ Bon |
| **GLB Standard** | `.glb`, `.gltf` | Divers | ⭐ Basique |

📖 **Guide complet** : [AVATARS_GUIDE.md](AVATARS_GUIDE.md)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Angular Application                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐      ┌──────────────┐      ┌──────────┐ │
│  │   Webcam      │─────▶│   MediaPipe  │─────▶│Kalidokit │ │
│  │   Input       │      │   Holistic   │      │ Solver   │ │
│  └───────────────┘      └──────────────┘      └─────┬────┘ │
│                                                      │       │
│                                                      ▼       │
│                         ┌──────────────────────────────┐    │
│                         │   AI Correction Service      │    │
│                         │   (ONNX Runtime - Optional)  │    │
│                         └──────────┬───────────────────┘    │
│                                    │                         │
│                                    ▼                         │
│                         ┌──────────────────────────────┐    │
│                         │   Animation Engine           │    │
│                         │   (Three.js AnimationMixer)  │    │
│                         └──────────┬───────────────────┘    │
│                                    │                         │
│                                    ▼                         │
│                         ┌──────────────────────────────┐    │
│                         │   3D Scene Renderer          │    │
│                         │   (Three.js WebGL)           │    │
│                         └──────────────────────────────┘    │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Modules Principaux

#### 1. **Tracking Service** (`tracking.service.ts`)
- Gère MediaPipe Holistic pour la détection des landmarks
- Traite les résultats avec Kalidokit
- Applique le lissage pour stabiliser les mouvements
- Calcule les métriques de performance (FPS, latence, qualité)

#### 2. **Animation Service** (`animation.service.ts`)
- Initialise la scène Three.js
- Charge et configure l'avatar 3D (GLB/FBX)
- Applique les transformations de tracking à l'avatar
- Gère les interactions avec les objets 3D via Raycaster

#### 3. **AI Correction Service** (`ai-correction.service.ts`)
- Infrastructure pour l'intégration de modèles PyTorch (ONNX)
- Lissage simple en attendant le modèle IA
- Prêt pour la prédiction et correction avancée

## 🛠️ Technologies

### Frontend
- **Angular 18+** - Framework applicatif
- **TypeScript 5.4+** - Langage principal
- **RxJS 7.8+** - Gestion d'état réactive

### Tracking & IA
- **MediaPipe Holistic 0.5+** - Détection des landmarks
- **Kalidokit 1.1+** - Conversion des landmarks en rotations
- **ONNX Runtime Web 1.17+** - Inférence IA côté client

### Rendu 3D
- **Three.js 0.160+** - Moteur de rendu WebGL
- **GLTFLoader** - Chargement de modèles 3D
- **OrbitControls** - Navigation dans la scène

### Outils de Développement
- **Angular CLI** - Outils de build
- **Node.js 18+** - Environnement d'exécution
- **VSCode** - Éditeur recommandé

## 📦 Installation

### Prérequis

```bash
Node.js >= 18.0.0
npm >= 9.0.0
```

### Étapes d'Installation

1. **Cloner le repository**
```bash
git clone https://github.com/Achraf-ghazouani/Angular_AvatarMotionTracking.git
cd Angular_AvatarMotionTracking
```

2. **Installer les dépendances**
```bash
npm install
```

3. **Créer le dossier des assets**
```bash
mkdir -p src/assets/models
mkdir -p src/assets/mediapipe
```

4. **Télécharger un modèle d'avatar (optionnel)**
- Télécharger un modèle GLB/FBX depuis [Ready Player Me](https://readyplayer.me/) ou [Mixamo](https://www.mixamo.com/)
- Placer le fichier dans `src/assets/models/avatar.glb`
- Si aucun modèle n'est fourni, un avatar de substitution sera créé automatiquement

5. **Lancer l'application**
```bash
npm start
```

6. **Accéder à l'application**
```
http://localhost:4200
```

## 🎯 Utilisation

### Démarrage Rapide

1. **Autoriser l'accès à la webcam** lorsque le navigateur le demande
2. **Cliquer sur "Start Tracking"** dans le panneau gauche
3. **Positionner votre visage** dans le cadre de la webcam
4. **Observer l'avatar** reproduire vos mouvements en temps réel

### Contrôles 3D

| Action | Commande |
|--------|----------|
| Rotation de la caméra | Clic gauche + glisser |
| Déplacement de la caméra | Clic droit + glisser |
| Zoom | Molette de la souris |
| Interaction avec le cube | Cliquer sur le cube rouge |

### Panneau de Performance

Le panneau affiche en temps réel :
- **FPS** : Images par seconde (objectif: ≥30)
- **Latency** : Temps de traitement (objectif: ≤100ms)
- **Quality** : Qualité du tracking (objectif: ≥90%)

## ⚙️ Configuration

### Configuration MediaPipe

Ajuster dans `src/app/models/config.model.ts` :

```typescript
mediapipe: {
  modelComplexity: 1,        // 0=Rapide, 1=Équilibré, 2=Précis
  smoothLandmarks: true,     // Lissage des landmarks
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
}
```

### Configuration Three.js

```typescript
threejs: {
  antialias: true,
  powerPreference: 'high-performance',
  alpha: true
}
```

### Configuration Avatar

```typescript
avatar: {
  modelPath: 'assets/models/avatar.glb',
  scale: 1,
  position: { x: 0, y: -1, z: 0 }
}
```

## 📈 Phases de Développement

### ✅ Phase 1 - Prototype (Terminée)
- Intégration MediaPipe Holistic
- Configuration Kalidokit
- Rendu 3D de base avec Three.js
- Avatar animé basique

**Critères de validation :**
- ✅ Tracking facial opérationnel
- ✅ Avatar se déplace avec l'utilisateur
- ✅ FPS ≥ 30

### ✅ Phase 2 - Stabilisation (Terminée)
- Implémentation du lissage des mouvements
- Correction des erreurs Kalidokit
- Optimisation des performances

**Critères de validation :**
- ✅ Mouvements fluides sans jitter
- ✅ Latence ≤ 100ms
- ✅ Fidélité des mouvements ≥ 90%

### 🚧 Phase 3 - IA (Infrastructure Prête)
- Structure pour modèle LSTM PyTorch
- Intégration ONNX Runtime
- Prédiction et correction intelligente

**Pour activer l'IA (voir section suivante)**

### ✅ Phase 4 - Interaction (Terminée)
- Raycaster pour sélection d'objets
- Cube interactif de test
- Système d'événements

**Critères de validation :**
- ✅ Cube cliquable et manipulable
- ✅ Feedback visuel sur interaction

### ✅ Phase 5 - Sécurité (Terminée)
- Traitement 100% frontend
- Aucun transfert de données
- Validation de la confidentialité

## 🤖 Intégration IA (Phase 3)

L'infrastructure pour l'IA est prête mais **désactivée par défaut**. Pour l'activer :

### 1. Entraîner un Modèle PyTorch

```python
import torch
import torch.nn as nn

class MotionCorrectionLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Entraînement du modèle
model = MotionCorrectionLSTM()
# ... votre boucle d'entraînement ...

# Export en ONNX
dummy_input = torch.randn(1, 10, 12)  # [batch, sequence, features]
torch.onnx.export(
    model, 
    dummy_input, 
    "motion_correction.onnx",
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch', 1: 'sequence'}}
)
```

### 2. Placer le Modèle

```bash
cp motion_correction.onnx src/assets/models/
```

### 3. Activer dans la Configuration

```typescript
// src/app/models/config.model.ts
ai: {
  enabled: true,
  modelPath: 'assets/models/motion_correction.onnx',
  inferenceType: 'onnx',
  smoothingFactor: 0.7,
  predictionSteps: 3
}
```

### Features d'Entrée du Modèle

Le modèle attend une séquence de 12 features par frame :
- **0-2** : Rotation de la tête (x, y, z)
- **3-5** : Rotation des hanches (x, y, z)
- **6-8** : Position du bras gauche (x, y, z)
- **9-11** : Position du bras droit (x, y, z)

## 📊 Performance

### Objectifs de Performance

| Métrique | Objectif | Critique |
|----------|----------|----------|
| FPS | ≥ 30 | ≥ 24 |
| Latence | ≤ 100ms | ≤ 150ms |
| Qualité Tracking | ≥ 90% | ≥ 70% |
| Bundle Size | < 20 MB | < 25 MB |

### Optimisations Implémentées

1. **Lissage adaptatif** - Réduit le jitter sans ajouter de latence
2. **Buffer circulaire** - Gestion efficace de la mémoire
3. **Lazy loading** - ONNX Runtime chargé uniquement si nécessaire
4. **WebGL optimisé** - Configuration Three.js haute performance
5. **Tree shaking** - Build optimisé Angular

### Navigateurs Supportés

| Navigateur | Version Minimale | Support WebGL 2 |
|------------|------------------|-----------------|
| Chrome | 90+ | ✅ |
| Edge | 90+ | ✅ |
| Firefox | 88+ | ✅ |
| Safari | 15+ | ⚠️ (Limited) |

## 📁 Structure du Projet

```
Angular_AvatarMotionTracking/
├── src/
│   ├── app/
│   │   ├── models/
│   │   │   ├── tracking.model.ts      # Types de données de tracking
│   │   │   └── config.model.ts        # Configuration globale
│   │   ├── services/
│   │   │   ├── tracking.service.ts    # MediaPipe + Kalidokit
│   │   │   ├── animation.service.ts   # Three.js rendering
│   │   │   └── ai-correction.service.ts # IA correction
│   │   ├── app.component.ts           # Composant principal
│   │   ├── app.component.html         # Template UI
│   │   └── app.component.scss         # Styles
│   ├── assets/
│   │   ├── models/
│   │   │   ├── avatar.glb            # Modèle 3D avatar
│   │   │   └── motion_correction.onnx # Modèle IA (optionnel)
│   │   └── mediapipe/
│   │       └── holistic/             # Fichiers MediaPipe
│   ├── index.html
│   ├── main.ts
│   └── styles.scss
├── angular.json
├── package.json
├── tsconfig.json
└── README.md
```

## 🔧 Développement

### Commandes Utiles

```bash
# Développement
npm start                 # Lancer le serveur dev
npm run build            # Build production
npm run watch            # Build avec watch mode

# Debugging
npm run lint             # Vérifier le code
```

### Debugging

Pour activer les logs détaillés :
```typescript
// Dans app.component.ts
console.log('🐛 Debug mode enabled');
```

## 🚀 Déploiement

### Build de Production

```bash
npm run build
```

Les fichiers seront générés dans `dist/avatar-motion-tracking/`.

### Contraintes de Déploiement

- ⚠️ **HTTPS requis** pour l'accès webcam
- ⚠️ **Headers CORS** nécessaires pour les fichiers MediaPipe
- ⚠️ **Bundle size** : Vérifier que le total reste < 20 MB

### Serveur Statique

```bash
# Exemple avec serve
npm install -g serve
serve -s dist/avatar-motion-tracking
```

## 📝 Livrables

- ✅ Application Angular complète et fonctionnelle
- ✅ Infrastructure pour modèle IA (ONNX)
- ✅ Support avatar GLB/FBX + fallback
- ✅ Documentation technique complète
- ✅ Rapport de performance intégré

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 License

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- **MediaPipe** - Google pour l'excellent framework de tracking
- **Kalidokit** - Pour la conversion des landmarks
- **Three.js** - Pour le rendu 3D WebGL
- **Angular Team** - Pour le framework robuste

## 📧 Contact

Achraf Ghazouani - [@Achraf-ghazouani](https://github.com/Achraf-ghazouani)

Project Link: [https://github.com/Achraf-ghazouani/Angular_AvatarMotionTracking](https://github.com/Achraf-ghazouani/Angular_AvatarMotionTracking)

---

**Made with ❤️ using Angular + Three.js + MediaPipe**