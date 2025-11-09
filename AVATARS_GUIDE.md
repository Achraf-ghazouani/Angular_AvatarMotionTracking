# 🎭 Guide d'Utilisation des Avatars

Ce guide vous explique comment utiliser différents types d'avatars avec votre application de motion tracking.

## 📋 Table des Matières

- [Formats Supportés](#formats-supportés)
- [Mixamo Avatars](#mixamo-avatars)
- [VRM Avatars](#vrm-avatars)
- [Configuration](#configuration)
- [Exemples](#exemples)

---

## Formats Supportés

L'application supporte **3 types d'avatars** :

### 1. **VRM** (.vrm) ⭐ Recommandé
- Format standard pour les avatars VTuber
- Bones humanoid standardisés
- Excellente compatibilité
- Support des expressions faciales

### 2. **Mixamo** (.fbx, .glb)
- Avatars rigged de Mixamo.com
- Bones automatiquement reconnus
- Grande bibliothèque gratuite
- Parfait pour débuter

### 3. **GLB/GLTF Standard** (.glb, .gltf)
- Format 3D générique
- Nécessite un rigging humanoid
- Compatible Three.js

---

## Mixamo Avatars

### Télécharger un Avatar Mixamo

1. **Aller sur [Mixamo.com](https://www.mixamo.com/)**
   ```
   - Créer un compte Adobe (gratuit)
   - Parcourir la bibliothèque "Characters"
   ```

2. **Choisir un Personnage**
   ```
   Recommandations:
   - Amy
   - Kaya
   - Remy
   - Megan
   ```

3. **Télécharger le Modèle**
   ```
   Format: FBX Binary ou GLB
   Pose: T-Pose (important !)
   Skin: With Skin
   
   Cliquer sur "Download"
   ```

4. **Placer dans le Projet**
   ```bash
   # Créer le dossier models
   mkdir src/assets/models
   
   # Copier le fichier téléchargé
   cp ~/Downloads/Kaya.fbx src/assets/models/avatar.fbx
   # ou
   cp ~/Downloads/Kaya.glb src/assets/models/avatar.glb
   ```

5. **Configurer dans l'Application**
   ```typescript
   // src/app/models/config.model.ts
   avatar: {
     modelPath: 'assets/models/avatar.fbx',  // ou .glb
     scale: 0.01,  // Mixamo est souvent grand (100x)
     position: { x: 0, y: 0, z: 0 },
     rotation: { x: 0, y: 0, z: 0 }
   }
   ```

### Ajustements Mixamo

Les avatars Mixamo nécessitent parfois des ajustements :

```typescript
// Avatar trop grand
scale: 0.01  // Réduit de 100x

// Avatar tourné dans le mauvais sens
rotation: { x: 0, y: Math.PI, z: 0 }  // Rotation 180°

// Avatar trop bas/haut
position: { x: 0, y: -1, z: 0 }
```

---

## VRM Avatars

### Télécharger un Avatar VRM

#### Option 1: VRoid Hub (Recommandé)

1. **Aller sur [VRoid Hub](https://hub.vroid.com/)**
   ```
   - Parcourir les avatars
   - Filtrer: "Downloadable" + "Commercial Use Allowed"
   ```

2. **Télécharger**
   ```
   - Choisir un avatar
   - Cliquer "Download"
   - Format: .vrm
   ```

#### Option 2: Créer avec VRoid Studio

1. **Télécharger [VRoid Studio](https://vroid.com/en/studio)**
   ```
   Gratuit pour Windows/Mac
   ```

2. **Créer votre Avatar**
   ```
   - Personnaliser l'apparence
   - Exporter en VRM
   ```

#### Option 3: Ready Player Me

1. **Créer sur [Ready Player Me](https://readyplayer.me/)**
   ```
   - Créer un avatar depuis une photo
   - Télécharger en GLB
   ```

### Placer l'Avatar VRM

```bash
# Copier le fichier VRM
cp ~/Downloads/my-avatar.vrm src/assets/models/avatar.vrm
```

### Configuration VRM

```typescript
// src/app/models/config.model.ts
avatar: {
  modelPath: 'assets/models/avatar.vrm',
  scale: 1,  // VRM sont déjà à la bonne échelle
  position: { x: 0, y: 0, z: 0 },
  rotation: { x: 0, y: 0, z: 0 }  // VRM sont auto-orientés
}
```

---

## Configuration

### Dans config.model.ts

```typescript
export const DEFAULT_CONFIG: AppConfig = {
  avatar: {
    modelPath: 'assets/models/avatar.vrm',  // Votre avatar
    scale: 1,
    position: { x: 0, y: 0, z: 0 },
    rotation: { x: 0, y: 0, z: 0 }
  },
  // ... autres configurations
};
```

### Dynamique (dans l'interface)

L'application détecte automatiquement le type d'avatar basé sur l'extension :

- `.vrm` → VRM loader
- `.fbx` → FBX loader (Mixamo)
- `.glb` / `.gltf` → GLTF loader

---

## Exemples

### Exemple 1: Avatar Mixamo (Kaya)

```typescript
avatar: {
  modelPath: 'assets/models/kaya.fbx',
  scale: 0.01,  // Mixamo scale
  position: { x: 0, y: -1, z: 0 },
  rotation: { x: 0, y: 0, z: 0 }
}
```

### Exemple 2: Avatar VRM

```typescript
avatar: {
  modelPath: 'assets/models/my-vtuber.vrm',
  scale: 1,
  position: { x: 0, y: 0, z: 0 },
  rotation: { x: 0, y: 0, z: 0 }
}
```

### Exemple 3: Ready Player Me (GLB)

```typescript
avatar: {
  modelPath: 'assets/models/readyplayerme.glb',
  scale: 2,  // RPM sont souvent petits
  position: { x: 0, y: -1.7, z: 0 },
  rotation: { x: 0, y: 0, z: 0 }
}
```

---

## Mapping des Bones

### Mixamo → Kalidokit

L'application mappe automatiquement les bones Mixamo :

```
mixamorigHead → Head
mixamorigNeck → Neck
mixamorigHips → Hips
mixamorigSpine → Spine
mixamorigLeftArm → LeftUpperArm
mixamorigLeftForeArm → LeftLowerArm
mixamorigLeftHand → LeftHand
... (voir avatar-loader.service.ts pour la liste complète)
```

### VRM Humanoid Bones

VRM utilise un système standardisé :
```
head, neck, hips, spine, chest
leftUpperArm, leftLowerArm, leftHand
rightUpperArm, rightLowerArm, rightHand
leftUpperLeg, leftLowerLeg, leftFoot
... (standard VRM)
```

---

## Dépannage

### ❌ L'avatar ne se charge pas

**Vérifier :**
1. Le chemin du fichier est correct
2. Le fichier est dans `src/assets/models/`
3. L'extension est supportée (.vrm, .fbx, .glb, .gltf)
4. Le fichier n'est pas corrompu

**Console :**
```javascript
F12 > Console
// Chercher les erreurs de chargement
```

### ❌ L'avatar est invisible

**Solutions :**
```typescript
// Ajuster l'échelle
scale: 0.01  // ou 0.1, 1, 10, 100

// Ajuster la position
position: { x: 0, y: -2, z: 0 }  // Descendre
position: { x: 0, y: 2, z: 0 }   // Monter

// Vérifier la caméra
camera.position.set(0, 1.6, 3);  // Reculer si nécessaire
```

### ❌ L'avatar ne bouge pas

**Vérifier :**
1. Le tracking fonctionne (webcam active)
2. Les bones sont détectés (voir console: "📊 Bones found: X")
3. Le rigging est correct (T-pose pour Mixamo)

**Debug :**
```typescript
// Dans la console
console.log(avatarInfo.bones);  // Voir les bones disponibles
```

### ❌ Les mouvements sont étranges

**Ajuster :**
```typescript
// Rotation incorrecte
rotation: { x: 0, y: Math.PI, z: 0 }  // Pivoter

// Échelle incorrecte
scale: 0.01  // Mixamo: 0.01
scale: 1     // VRM: 1
scale: 2     // RPM: 2
```

---

## Resources Gratuites

### Avatars Mixamo
- **Site :** https://www.mixamo.com/
- **Licence :** Gratuit avec compte Adobe
- **Formats :** FBX, GLB
- **Quantité :** 100+ personnages

### VRoid Hub
- **Site :** https://hub.vroid.com/
- **Licence :** Variable (vérifier par avatar)
- **Formats :** VRM
- **Quantité :** Des milliers

### Ready Player Me
- **Site :** https://readyplayer.me/
- **Licence :** Gratuit
- **Formats :** GLB
- **Quantité :** Illimité (générateur)

### The Base Mesh
- **Site :** https://thebasemesh.com/
- **Licence :** Gratuit/Payant
- **Formats :** FBX, OBJ
- **Quantité :** Bibliothèque variée

---

## Structure des Fichiers

```
src/
└── assets/
    └── models/
        ├── avatar.vrm       # Avatar VRM principal
        ├── kaya.fbx         # Avatar Mixamo
        ├── custom.glb       # Avatar GLB custom
        └── fallback.glb     # Avatar de secours
```

---

## Checklist de Configuration

- [ ] Avatar téléchargé
- [ ] Fichier placé dans `src/assets/models/`
- [ ] Chemin configuré dans `config.model.ts`
- [ ] Échelle ajustée
- [ ] Position ajustée
- [ ] Rotation ajustée (si nécessaire)
- [ ] Application rebuild (`npm start`)
- [ ] Test du tracking

---

## Support

**Problèmes courants :** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**Documentation complète :** [README.md](README.md)

**Configuration exemples :** [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)

---

**Bon tracking ! 🎭✨**
