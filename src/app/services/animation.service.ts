import { Injectable } from '@angular/core';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VRM, VRMUtils } from '@pixiv/three-vrm';
import { BehaviorSubject, Observable } from 'rxjs';
import { KalidoKitResults } from '@models/tracking.model';
import { ThreeJsConfig, AvatarConfig } from '@models/config.model';
import { AvatarLoaderService, AvatarInfo } from './avatar-loader.service';

/**
 * Service de rendu 3D et d'animation avec Three.js
 * Phase 1 & 4: Animation de l'avatar et interaction avec objets
 * Supporte: VRM, Mixamo (FBX/GLB), GLB standard
 */
@Injectable({
  providedIn: 'root'
})
export class AnimationService {
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private animationMixer?: THREE.AnimationMixer;
  private clock = new THREE.Clock();

  // Avatar et objets
  private avatarInfo?: AvatarInfo;
  private testCube?: THREE.Mesh;
  private interactiveObjects: THREE.Object3D[] = [];

  // Raycasting pour interaction
  private raycaster = new THREE.Raycaster();
  private mouse = new THREE.Vector2();
  private selectedObject: THREE.Object3D | null = null;

  // État
  private isInitialized$ = new BehaviorSubject<boolean>(false);
  private isAvatarLoaded$ = new BehaviorSubject<boolean>(false);
  private currentAnimation$ = new BehaviorSubject<string>('idle');

  // Animation loop
  private animationFrameId?: number;

  constructor(private avatarLoader: AvatarLoaderService) {}

  /**
   * Initialise la scène Three.js
   */
  async initialize(container: HTMLElement, config: ThreeJsConfig): Promise<void> {
    // Créer la scène
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x1a1a2e);
    this.scene.fog = new THREE.Fog(0x1a1a2e, 10, 50);

    // Créer la caméra
    this.camera = new THREE.PerspectiveCamera(
      45,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 1.6, 3);

    // Créer le renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: config.antialias,
      alpha: config.alpha,
      powerPreference: config.powerPreference,
      preserveDrawingBuffer: config.preserveDrawingBuffer
    });
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    
    container.appendChild(this.renderer.domElement);

    // Ajouter les contrôles
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 1, 0);
    this.controls.update();

    // Ajouter les lumières
    this.setupLights();

    // Ajouter le sol
    this.addGround();

    // Phase 4: Ajouter le cube interactif
    this.addTestCube();

    // Gérer le redimensionnement
    window.addEventListener('resize', () => this.onWindowResize(container));

    // Gérer les événements de souris pour l'interaction
    this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
    this.renderer.domElement.addEventListener('click', (e) => this.onMouseClick(e));

    this.isInitialized$.next(true);
    console.log('✅ Three.js scene initialized');
  }

  /**
   * Configure l'éclairage de la scène
   */
  private setupLights(): void {
    // Lumière ambiante
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(ambientLight);

    // Lumière directionnelle principale
    const mainLight = new THREE.DirectionalLight(0xffffff, 0.8);
    mainLight.position.set(5, 10, 5);
    mainLight.castShadow = true;
    mainLight.shadow.mapSize.width = 2048;
    mainLight.shadow.mapSize.height = 2048;
    mainLight.shadow.camera.near = 0.5;
    mainLight.shadow.camera.far = 50;
    this.scene.add(mainLight);

    // Lumière de remplissage
    const fillLight = new THREE.DirectionalLight(0x8899ff, 0.3);
    fillLight.position.set(-5, 5, -5);
    this.scene.add(fillLight);

    // Lumière arrière
    const backLight = new THREE.DirectionalLight(0xffffff, 0.2);
    backLight.position.set(0, 5, -10);
    this.scene.add(backLight);
  }

  /**
   * Ajoute un sol à la scène
   */
  private addGround(): void {
    const groundGeometry = new THREE.PlaneGeometry(20, 20);
    const groundMaterial = new THREE.MeshStandardMaterial({
      color: 0x16213e,
      roughness: 0.8,
      metalness: 0.2
    });
    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    this.scene.add(ground);

    // Ajouter une grille
    const gridHelper = new THREE.GridHelper(20, 20, 0x0f3460, 0x0f3460);
    this.scene.add(gridHelper);
  }

  /**
   * Phase 4: Ajoute un cube interactif pour les tests
   */
  private addTestCube(): void {
    const geometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
    const material = new THREE.MeshStandardMaterial({
      color: 0xe94560,
      roughness: 0.4,
      metalness: 0.6,
      emissive: 0xe94560,
      emissiveIntensity: 0.2
    });
    
    this.testCube = new THREE.Mesh(geometry, material);
    this.testCube.position.set(1.5, 0.25, 0);
    this.testCube.castShadow = true;
    this.testCube.receiveShadow = true;
    this.testCube.userData = { interactive: true, type: 'cube' };
    
    this.scene.add(this.testCube);
    this.interactiveObjects.push(this.testCube);

    console.log('✅ Interactive test cube added');
  }

  /**
   * Charge l'avatar (VRM, Mixamo FBX/GLB, ou GLB standard)
   */
  async loadAvatar(config: AvatarConfig): Promise<void> {
    try {
      // Supprimer l'ancien avatar s'il existe
      this.removeCurrentAvatar();

      console.log('📦 Loading avatar from:', config.modelPath);
      console.log('📐 Config:', { scale: config.scale, position: config.position, rotation: config.rotation });

      // Charger l'avatar avec le service approprié
      this.avatarInfo = await this.avatarLoader.loadAvatar(config.modelPath, config);

      let avatarObject: THREE.Object3D;

      // Ajouter à la scène
      if (this.avatarInfo.isVRM) {
        const vrm = this.avatarInfo.object as VRM;
        avatarObject = vrm.scene;
        this.scene.add(vrm.scene);
        console.log('✅ VRM avatar added to scene');
      } else {
        avatarObject = this.avatarInfo.object as THREE.Object3D;
        this.scene.add(avatarObject);
        console.log('✅ Non-VRM avatar added to scene');
      }

      // S'assurer que l'avatar est visible
      avatarObject.visible = true;
      avatarObject.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          child.visible = true;
          child.castShadow = true;
          child.receiveShadow = true;
        }
      });

      // Calculer et afficher les dimensions de l'avatar
      const bbox = new THREE.Box3().setFromObject(avatarObject);
      const size = new THREE.Vector3();
      bbox.getSize(size);
      const center = new THREE.Vector3();
      bbox.getCenter(center);

      console.log('📏 Avatar dimensions:', {
        width: size.x.toFixed(2),
        height: size.y.toFixed(2),
        depth: size.z.toFixed(2)
      });
      console.log('📍 Avatar bounding box:', {
        min: `(${bbox.min.x.toFixed(2)}, ${bbox.min.y.toFixed(2)}, ${bbox.min.z.toFixed(2)})`,
        max: `(${bbox.max.x.toFixed(2)}, ${bbox.max.y.toFixed(2)}, ${bbox.max.z.toFixed(2)})`
      });
      console.log('📍 Avatar center:', {
        x: center.x.toFixed(2),
        y: center.y.toFixed(2),
        z: center.z.toFixed(2)
      });
      console.log('📍 Avatar position (before centering):', {
        x: avatarObject.position.x.toFixed(2),
        y: avatarObject.position.y.toFixed(2),
        z: avatarObject.position.z.toFixed(2)
      });
      console.log('📍 Avatar scale:', {
        x: avatarObject.scale.x.toFixed(2),
        y: avatarObject.scale.y.toFixed(2),
        z: avatarObject.scale.z.toFixed(2)
      });

      // Centrer l'avatar si nécessaire
      this.centerAvatar(avatarObject, bbox);

      this.isAvatarLoaded$.next(true);
      
      console.log(`✅ Avatar loaded successfully (type: ${this.avatarInfo.type})`);
      console.log(`📊 Bones found: ${this.avatarInfo.bones.size}`);
    } catch (error) {
      console.error('❌ Error loading avatar:', error);
      
      // Ne PAS créer d'avatar de substitution
      // L'utilisateur doit charger son propre avatar
      console.log('⚠️ No avatar loaded. Please upload a valid avatar file.');
      throw error; // Propager l'erreur pour que l'appelant puisse la gérer
    }
  }

  /**
   * Centre l'avatar dans la scène
   */
  private centerAvatar(avatar: THREE.Object3D, bbox: THREE.Box3): void {
    const size = new THREE.Vector3();
    bbox.getSize(size);
    const center = new THREE.Vector3();
    bbox.getCenter(center);

    // Ajuster la position Y pour que l'avatar soit au sol
    // Soustraire la moitié de la hauteur et ajouter le centre Y
    const groundOffset = -bbox.min.y;
    
    console.log('🎯 Centering avatar...');
    console.log('   Ground offset:', groundOffset.toFixed(2));
    
    // Appliquer l'offset au sol
    avatar.position.y += groundOffset;

    // Centrer sur X et Z
    avatar.position.x -= center.x;
    avatar.position.z -= center.z;

    console.log('📍 New avatar position:', {
      x: avatar.position.x.toFixed(2),
      y: avatar.position.y.toFixed(2),
      z: avatar.position.z.toFixed(2)
    });

    // Ajuster la caméra pour voir l'avatar entier
    const maxDim = Math.max(size.x, size.y, size.z);
    const fov = this.camera.fov * (Math.PI / 180);
    let cameraDistance = Math.abs(maxDim / Math.sin(fov / 2));
    cameraDistance *= 1.5; // Ajouter un peu de marge

    this.camera.position.set(0, size.y * 0.5, cameraDistance);
    this.camera.lookAt(0, size.y * 0.5, 0);
    
    // Mettre à jour les contrôles pour pointer vers le centre de l'avatar
    if (this.controls) {
      this.controls.target.set(0, size.y * 0.5, 0);
      this.controls.update();
    }

    console.log('📷 Camera adjusted to distance:', cameraDistance.toFixed(2));
    console.log('📷 Camera position:', {
      x: this.camera.position.x.toFixed(2),
      y: this.camera.position.y.toFixed(2),
      z: this.camera.position.z.toFixed(2)
    });
    console.log('📷 Camera target:', {
      x: this.controls?.target.x.toFixed(2) ?? '0',
      y: this.controls?.target.y.toFixed(2) ?? '0',
      z: this.controls?.target.z.toFixed(2) ?? '0'
    });
    this.controls.target.set(0, size.y * 0.5, 0);
    this.controls.update();

    console.log('📷 Camera adjusted to distance:', cameraDistance.toFixed(2));
  }

  /**
   * Supprime l'avatar actuel de la scène
   */
  private removeCurrentAvatar(): void {
    if (!this.avatarInfo) return;

    try {
      if (this.avatarInfo.isVRM) {
        const vrm = this.avatarInfo.object as VRM;
        this.scene.remove(vrm.scene);
        
        // Nettoyer les ressources VRM
        VRMUtils.deepDispose(vrm.scene);
      } else {
        const object = this.avatarInfo.object as THREE.Object3D;
        this.scene.remove(object);
        
        // Nettoyer les géométries et matériaux
        object.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            if (mesh.geometry) mesh.geometry.dispose();
            if (mesh.material) {
              if (Array.isArray(mesh.material)) {
                mesh.material.forEach(mat => mat.dispose());
              } else {
                mesh.material.dispose();
              }
            }
          }
        });
      }

      console.log('🗑️ Previous avatar removed');
    } catch (error) {
      console.warn('⚠️ Error removing previous avatar:', error);
    }
  }

  /**
   * Applique les résultats du tracking à l'avatar
   */
  applyTrackingToAvatar(results: KalidoKitResults): void {
    if (!this.avatarInfo) return;

    try {
      if (this.avatarInfo.isVRM) {
        // Appliquer le tracking à un avatar VRM
        const vrm = this.avatarInfo.object as VRM;
        this.avatarLoader.applyTrackingToVRM(vrm, results);
      } else {
        // Appliquer le tracking à un avatar Mixamo/GLB
        this.avatarLoader.applyTrackingToMixamo(this.avatarInfo.bones, results);
      }
    } catch (error) {
      console.warn('⚠️ Error applying tracking to avatar:', error);
    }
  }

  /**
   * Phase 4: Gestion du mouvement de la souris
   */
  private onMouseMove(event: MouseEvent): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Mettre à jour le raycaster
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.interactiveObjects, true);

    // Changer le curseur si on survole un objet
    if (intersects.length > 0) {
      this.renderer.domElement.style.cursor = 'pointer';
      
      // Highlight effect
      intersects[0].object.traverse((child) => {
        if ((child as THREE.Mesh).isMesh) {
          const mesh = child as THREE.Mesh;
          const material = mesh.material as THREE.MeshStandardMaterial;
          if (material.emissiveIntensity !== undefined) {
            material.emissiveIntensity = 0.5;
          }
        }
      });
    } else {
      this.renderer.domElement.style.cursor = 'default';
      
      // Reset highlight
      this.interactiveObjects.forEach(obj => {
        obj.traverse((child) => {
          if ((child as THREE.Mesh).isMesh) {
            const mesh = child as THREE.Mesh;
            const material = mesh.material as THREE.MeshStandardMaterial;
            if (material.emissiveIntensity !== undefined) {
              material.emissiveIntensity = 0.2;
            }
          }
        });
      });
    }
  }

  /**
   * Phase 4: Gestion du clic
   */
  private onMouseClick(event: MouseEvent): void {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.interactiveObjects, true);

    if (intersects.length > 0) {
      const object = intersects[0].object;
      this.selectedObject = object;
      
      console.log('🖱️ Object clicked:', object.userData);
      
      // Animation de clic
      const originalScale = object.scale.clone();
      object.scale.multiplyScalar(0.9);
      
      setTimeout(() => {
        object.scale.copy(originalScale);
      }, 100);
    }
  }

  /**
   * Démarre la boucle d'animation
   */
  startAnimation(): void {
    const animate = () => {
      this.animationFrameId = requestAnimationFrame(animate);

      const delta = this.clock.getDelta();

      // Mettre à jour les contrôles
      this.controls.update();

      // Mettre à jour l'animation mixer
      if (this.animationMixer) {
        this.animationMixer.update(delta);
      }

      // Rotation du cube pour l'effet visuel
      if (this.testCube) {
        this.testCube.rotation.y += delta * 0.5;
      }

      // Rendu
      this.renderer.render(this.scene, this.camera);
    };

    animate();
  }

  /**
   * Arrête la boucle d'animation
   */
  stopAnimation(): void {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = undefined;
    }
  }

  /**
   * Gestion du redimensionnement
   */
  private onWindowResize(container: HTMLElement): void {
    this.camera.aspect = container.clientWidth / container.clientHeight;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(container.clientWidth, container.clientHeight);
  }

  /**
   * Observables publics
   */
  getInitializedState(): Observable<boolean> {
    return this.isInitialized$.asObservable();
  }

  getAvatarLoadedState(): Observable<boolean> {
    return this.isAvatarLoaded$.asObservable();
  }

  getCurrentAnimation(): Observable<string> {
    return this.currentAnimation$.asObservable();
  }

  /**
   * Obtient les informations de l'avatar actuel
   */
  getCurrentAvatarInfo(): AvatarInfo | undefined {
    return this.avatarInfo;
  }

  /**
   * Observable pour les infos de l'avatar
   */
  getAvatarInfo(): Observable<AvatarInfo | undefined> {
    return new BehaviorSubject<AvatarInfo | undefined>(this.avatarInfo).asObservable();
  }

  /**
   * Ajuste l'échelle de l'avatar
   */
  setAvatarScale(scale: number): void {
    if (!this.avatarInfo) return;

    if (this.avatarInfo.isVRM) {
      const vrm = this.avatarInfo.object as VRM;
      vrm.scene.scale.setScalar(scale);
    } else {
      const object = this.avatarInfo.object as THREE.Object3D;
      object.scale.setScalar(scale);
    }

    console.log('📏 Avatar scale set to:', scale);
  }

  /**
   * Recentre l'avatar dans la scène
   */
  recenterAvatar(): void {
    if (!this.avatarInfo) return;

    const avatarObject = this.avatarInfo.isVRM 
      ? (this.avatarInfo.object as VRM).scene 
      : (this.avatarInfo.object as THREE.Object3D);

    const bbox = new THREE.Box3().setFromObject(avatarObject);
    this.centerAvatar(avatarObject, bbox);
  }

  /**
   * Obtenir la scène pour debug
   */
  getScene(): THREE.Scene {
    return this.scene;
  }

  /**
   * Nettoyage
   */
  destroy(): void {
    this.stopAnimation();

    // Nettoyer les événements
    window.removeEventListener('resize', () => this.onWindowResize);
    
    // Nettoyer la scène
    this.scene.traverse((object) => {
      if ((object as THREE.Mesh).isMesh) {
        const mesh = object as THREE.Mesh;
        mesh.geometry.dispose();
        if (Array.isArray(mesh.material)) {
          mesh.material.forEach(material => material.dispose());
        } else {
          mesh.material.dispose();
        }
      }
    });

    // Dispose du renderer
    this.renderer.dispose();
    this.controls.dispose();

    console.log('🧹 Three.js scene cleaned up');
  }
}
