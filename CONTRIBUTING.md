# Guide de Contribution

Merci de votre intérêt pour contribuer à **Avatar IA Motion Tracking** ! 🎉

## 📋 Comment Contribuer

### 1. Types de Contributions

Nous acceptons les contributions suivantes :
- 🐛 **Bug fixes** - Corrections de bugs
- ✨ **Nouvelles fonctionnalités** - Nouvelles features
- 📚 **Documentation** - Amélioration de la doc
- 🎨 **Design/UI** - Améliorations visuelles
- ⚡ **Performance** - Optimisations
- 🧪 **Tests** - Ajout de tests unitaires/e2e

### 2. Processus de Contribution

#### Étape 1 : Fork & Clone
```bash
# Fork le repository sur GitHub
# Puis cloner votre fork
git clone https://github.com/VOTRE_USERNAME/Angular_AvatarMotionTracking.git
cd Angular_AvatarMotionTracking
```

#### Étape 2 : Créer une Branche
```bash
# Créer une branche pour votre feature
git checkout -b feature/ma-nouvelle-feature

# Ou pour un bug fix
git checkout -b fix/correction-bug
```

#### Étape 3 : Développer
```bash
# Installer les dépendances
npm install

# Lancer en mode développement
npm start

# Faire vos modifications...
```

#### Étape 4 : Tester
```bash
# Vérifier que l'application fonctionne
npm start

# (À venir) Lancer les tests
# npm test
```

#### Étape 5 : Commit
```bash
# Ajouter les fichiers modifiés
git add .

# Commit avec un message clair
git commit -m "✨ feat: Ajout de la feature X"

# Ou pour un bug fix
git commit -m "🐛 fix: Correction du bug Y"
```

**Format des messages de commit :**
- ✨ `feat:` - Nouvelle fonctionnalité
- 🐛 `fix:` - Correction de bug
- 📚 `docs:` - Documentation
- 🎨 `style:` - Formatage, style
- ♻️ `refactor:` - Refactoring
- ⚡ `perf:` - Performance
- 🧪 `test:` - Tests
- 🔧 `chore:` - Maintenance

#### Étape 6 : Push & Pull Request
```bash
# Push vers votre fork
git push origin feature/ma-nouvelle-feature

# Créer une Pull Request sur GitHub
```

### 3. Standards de Code

#### TypeScript
```typescript
// ✅ Bon - Types explicites
function processData(input: string): number {
  return parseInt(input, 10);
}

// ❌ Mauvais - Types implicites
function processData(input) {
  return parseInt(input);
}
```

#### Nommage
```typescript
// Classes - PascalCase
class TrackingService { }

// Méthodes/Fonctions - camelCase
startTracking() { }

// Constantes - UPPER_SNAKE_CASE
const MAX_BUFFER_SIZE = 10;

// Interfaces - PascalCase avec I prefix (optionnel)
interface TrackingState { }
```

#### Documentation
```typescript
/**
 * Description de la fonction
 * @param input - Description du paramètre
 * @returns Description du retour
 */
function myFunction(input: string): number {
  // ...
}
```

### 4. Structure des Fichiers

```
src/app/
├── models/          # Types et interfaces
├── services/        # Services Angular
├── components/      # Composants Angular
└── utils/           # Fonctions utilitaires
```

### 5. Pull Request Checklist

Avant de soumettre une PR, vérifier que :

- [ ] Le code compile sans erreur
- [ ] L'application fonctionne correctement
- [ ] Le code respecte les standards du projet
- [ ] Les commentaires sont à jour
- [ ] Le CHANGELOG.md est mis à jour
- [ ] La documentation est mise à jour si nécessaire
- [ ] Les tests passent (quand disponibles)
- [ ] Pas de `console.log()` oubliés
- [ ] Pas de code commenté inutile

### 6. Template de Pull Request

```markdown
## Description
[Description claire de ce que fait votre PR]

## Type de changement
- [ ] Bug fix
- [ ] Nouvelle fonctionnalité
- [ ] Breaking change
- [ ] Documentation

## Motivation
[Pourquoi ce changement est nécessaire]

## Tests effectués
- [ ] Test manuel
- [ ] Test sur Chrome
- [ ] Test sur Firefox
- [ ] Test sur Edge

## Screenshots
[Si applicable, ajouter des captures d'écran]

## Checklist
- [ ] Code testé
- [ ] Documentation mise à jour
- [ ] CHANGELOG.md mis à jour
```

## 🎯 Idées de Contributions

### Faciles (Good First Issue)
- 📝 Améliorer la documentation
- 🌐 Ajouter des traductions
- 🎨 Améliorer le design UI
- 🐛 Corriger des bugs mineurs

### Moyennes
- ✨ Ajouter de nouveaux avatars
- 📊 Améliorer les graphiques de performance
- 🎮 Ajouter de nouveaux objets interactifs
- ⚙️ Ajouter plus d'options de configuration

### Avancées
- 🤖 Améliorer le modèle IA
- 📹 Ajouter l'enregistrement vidéo
- 🖐️ Améliorer le tracking des mains
- 🎭 Ajouter le tracking des expressions faciales
- 📱 Optimisation mobile

## 🚫 Ce que nous n'acceptons PAS

- Code non testé
- Code sans documentation
- Dépendances inutiles ou lourdes
- Breaking changes sans discussion préalable
- Code non formaté
- Commits avec des messages vagues ("fix", "update", etc.)

## 🐛 Signaler un Bug

### Template de Bug Report

```markdown
**Décrire le bug**
[Description claire du bug]

**Reproduire le bug**
1. Aller sur '...'
2. Cliquer sur '...'
3. Voir l'erreur

**Comportement attendu**
[Ce qui devrait se passer]

**Screenshots**
[Si applicable]

**Environnement**
- OS: [Windows 11, macOS, Linux]
- Navigateur: [Chrome 120, Firefox 115, etc.]
- Version Node.js: [18.0.0]
- Version de l'application: [1.0.0]

**Logs/Erreurs**
```
[Copier les logs de la console]
```

**Informations additionnelles**
[Tout autre contexte utile]
```

## 💡 Proposer une Feature

### Template de Feature Request

```markdown
**La feature répond à quel problème ?**
[Description du problème]

**Solution proposée**
[Comment vous imaginez la feature]

**Alternatives considérées**
[Autres approches possibles]

**Contexte additionnel**
[Screenshots, mockups, etc.]
```

## 📚 Ressources

### Documentation Technique
- [Angular](https://angular.io/docs)
- [Three.js](https://threejs.org/docs/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [Kalidokit](https://github.com/yeemachine/kalidokit)
- [ONNX Runtime](https://onnxruntime.ai/docs/)

### Outils Recommandés
- [VSCode](https://code.visualstudio.com/) - Éditeur
- [Angular DevTools](https://angular.io/guide/devtools) - Debug
- [Chrome DevTools](https://developer.chrome.com/docs/devtools/) - Debug

## 👥 Équipe de Review

Les Pull Requests sont reviewées par :
- @Achraf-ghazouani - Mainteneur principal

Temps de réponse habituel : 2-7 jours

## 📜 Code of Conduct

### Notre Engagement

Nous nous engageons à faire de la participation à ce projet une expérience sans harcèlement pour tous.

### Standards

Exemples de comportements encouragés :
- ✅ Utiliser un langage accueillant et inclusif
- ✅ Respecter les points de vue différents
- ✅ Accepter les critiques constructives
- ✅ Se concentrer sur ce qui est mieux pour la communauté

Exemples de comportements inacceptables :
- ❌ Langage ou images sexualisés
- ❌ Trolling, commentaires insultants
- ❌ Harcèlement public ou privé
- ❌ Publication d'informations privées

## 🎓 Apprendre en Contribuant

Si vous êtes nouveau dans le projet :

1. **Commencer petit** - Documentation, typos, etc.
2. **Lire le code existant** - Comprendre l'architecture
3. **Poser des questions** - Via les Issues
4. **Proposer avant d'implémenter** - Discussion sur les features majeures

## 📬 Contact

- **Issues GitHub** : Pour bugs et features
- **Email** : achraf.ghazouani@example.com
- **Discussions** : GitHub Discussions (à venir)

## 🙏 Remerciements

Merci à tous les contributeurs actuels et futurs !

Votre temps et vos efforts sont grandement appréciés. 💖

---

**Happy Coding! 🚀**
