# 📁 Load Avatar from UI - Quick Guide

## ✨ Nouvelle Fonctionnalité

Vous pouvez maintenant **charger des avatars directement depuis l'interface** sans modifier le code !

## 🎯 Comment Utiliser

### Méthode 1: Charger un fichier local

1. **Cliquez sur le bouton "Load Avatar"** dans le panneau latéral gauche
2. **Sélectionnez votre fichier avatar** :
   - `.vrm` - Avatars VRM (VRoid, VTuber)
   - `.fbx` - Avatars Mixamo
   - `.glb` / `.gltf` - Avatars 3D standard
3. **L'avatar se charge automatiquement** et remplace l'ancien
4. **Le tracking s'applique immédiatement** 🎭

### Méthode 2: Preset rapide

- Cliquez sur **"Default"** pour charger l'avatar par défaut

## 📊 Informations Affichées

Le panneau Avatar affiche :
- **Type d'avatar** : VRM, MIXAMO, GLB
- **Nombre de bones** détectés
- **Statut** : Chargé / Non chargé

## 🔧 Échelle Automatique

L'application ajuste automatiquement l'échelle selon le format :
- **VRM** → Scale 1.0
- **FBX (Mixamo)** → Scale 0.01 (Mixamo est 100x trop grand)
- **GLB** → Scale 1.0

## ⚠️ Important

- **Arrêtez le tracking** avant de charger un nouvel avatar
- Le bouton "Load Avatar" est désactivé pendant le tracking
- L'ancien avatar est automatiquement supprimé

## 🎁 Où Trouver des Avatars

### VRM
- **VRoid Hub** : https://hub.vroid.com/
- **VRoid Studio** : https://vroid.com/studio (créer le vôtre)

### Mixamo
- **Mixamo.com** : https://www.mixamo.com/ (gratuit avec compte Adobe)
  - Format: FBX ou GLB
  - Pose: T-Pose

### Ready Player Me
- **readyplayer.me** : https://readyplayer.me/ (créer depuis une photo)

## 💡 Exemples

### Charger un Avatar VRM
```
1. Télécharger un .vrm depuis VRoid Hub
2. Cliquer "Load Avatar"
3. Sélectionner le fichier .vrm
4. ✅ C'est chargé !
```

### Charger un Avatar Mixamo
```
1. Télécharger Kaya.fbx depuis Mixamo
2. Cliquer "Load Avatar"
3. Sélectionner Kaya.fbx
4. ✅ L'avatar Mixamo apparaît !
```

## 🎨 Personnalisation Avancée

Pour ajuster manuellement l'échelle ou la position, modifiez le code :

```typescript
// src/app/app.component.ts - Méthode getDefaultScaleForType()
private getDefaultScaleForType(extension: string): number {
  switch (extension) {
    case 'fbx':
      return 0.01;  // Ajuster si nécessaire
    case 'vrm':
      return 1;
    case 'glb':
      return 1;     // Ou 2 pour Ready Player Me
    default:
      return 1;
  }
}
```

## 📚 Documentation Complète

- **Guide Avatars** : [AVATARS_GUIDE.md](AVATARS_GUIDE.md)
- **Dépannage** : [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Exemples de config** : [CONFIGURATION_EXAMPLES.md](CONFIGURATION_EXAMPLES.md)

---

**Profitez du motion tracking avec vos propres avatars ! 🎭✨**
