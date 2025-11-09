import { Injectable } from '@angular/core';
import * as THREE from 'three';
import { GLTFLoader, GLTF } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import { VRM, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { KalidoKitResults } from '@models/tracking.model';
import { AvatarConfig } from '@models/config.model';

export type AvatarType = 'vrm' | 'mixamo' | 'glb' | 'fbx';

export interface AvatarInfo {
  type: AvatarType;
  object: THREE.Object3D | VRM;
  bones: Map<string, THREE.Bone>;
  isVRM: boolean;
}

/**
 * Service pour charger différents types d'avatars
 * Supporte: VRM, Mixamo (FBX/GLB), GLB standard
 */
@Injectable({
  providedIn: 'root'
})
export class AvatarLoaderService {
  private gltfLoader!: GLTFLoader;
  private fbxLoader!: FBXLoader;

  // Mapping des bones Mixamo vers Kalidokit
  private readonly MIXAMO_BONE_MAP: { [key: string]: string } = {
    // Head
    'mixamorigHead': 'Head',
    'mixamorigNeck': 'Neck',
    
    // Spine
    'mixamorigSpine': 'Spine',
    'mixamorigSpine1': 'Chest',
    'mixamorigSpine2': 'UpperChest',
    
    // Hips
    'mixamorigHips': 'Hips',
    
    // Left Arm
    'mixamorigLeftShoulder': 'LeftShoulder',
    'mixamorigLeftArm': 'LeftUpperArm',
    'mixamorigLeftForeArm': 'LeftLowerArm',
    'mixamorigLeftHand': 'LeftHand',
    
    // Right Arm
    'mixamorigRightShoulder': 'RightShoulder',
    'mixamorigRightArm': 'RightUpperArm',
    'mixamorigRightForeArm': 'RightLowerArm',
    'mixamorigRightHand': 'RightHand',
    
    // Left Leg
    'mixamorigLeftUpLeg': 'LeftUpperLeg',
    'mixamorigLeftLeg': 'LeftLowerLeg',
    'mixamorigLeftFoot': 'LeftFoot',
    
    // Right Leg
    'mixamorigRightUpLeg': 'RightUpperLeg',
    'mixamorigRightLeg': 'RightLowerLeg',
    'mixamorigRightFoot': 'RightFoot',
    
    // Left Hand Fingers
    'mixamorigLeftHandThumb1': 'LeftThumbProximal',
    'mixamorigLeftHandThumb2': 'LeftThumbIntermediate',
    'mixamorigLeftHandThumb3': 'LeftThumbDistal',
    'mixamorigLeftHandIndex1': 'LeftIndexProximal',
    'mixamorigLeftHandIndex2': 'LeftIndexIntermediate',
    'mixamorigLeftHandIndex3': 'LeftIndexDistal',
    'mixamorigLeftHandMiddle1': 'LeftMiddleProximal',
    'mixamorigLeftHandMiddle2': 'LeftMiddleIntermediate',
    'mixamorigLeftHandMiddle3': 'LeftMiddleDistal',
    'mixamorigLeftHandRing1': 'LeftRingProximal',
    'mixamorigLeftHandRing2': 'LeftRingIntermediate',
    'mixamorigLeftHandRing3': 'LeftRingDistal',
    'mixamorigLeftHandPinky1': 'LeftLittleProximal',
    'mixamorigLeftHandPinky2': 'LeftLittleIntermediate',
    'mixamorigLeftHandPinky3': 'LeftLittleDistal',
    
    // Right Hand Fingers
    'mixamorigRightHandThumb1': 'RightThumbProximal',
    'mixamorigRightHandThumb2': 'RightThumbIntermediate',
    'mixamorigRightHandThumb3': 'RightThumbDistal',
    'mixamorigRightHandIndex1': 'RightIndexProximal',
    'mixamorigRightHandIndex2': 'RightIndexIntermediate',
    'mixamorigRightHandIndex3': 'RightIndexDistal',
    'mixamorigRightHandMiddle1': 'RightMiddleProximal',
    'mixamorigRightHandMiddle2': 'RightMiddleIntermediate',
    'mixamorigRightHandMiddle3': 'RightMiddleDistal',
    'mixamorigRightHandRing1': 'RightRingProximal',
    'mixamorigRightHandRing2': 'RightRingIntermediate',
    'mixamorigRightHandRing3': 'RightRingDistal',
    'mixamorigRightHandPinky1': 'RightLittleProximal',
    'mixamorigRightHandPinky2': 'RightLittleIntermediate',
    'mixamorigRightHandPinky3': 'RightLittleDistal'
  };

  constructor() {
    this.initializeLoaders();
  }

  /**
   * Initialise les loaders
   */
  private initializeLoaders(): void {
    this.gltfLoader = new GLTFLoader();
    
    // Ajouter le plugin VRM au GLTFLoader
    this.gltfLoader.register((parser) => {
      return new VRMLoaderPlugin(parser);
    });

    this.fbxLoader = new FBXLoader();
  }

  /**
   * Charge un avatar (détection automatique du type)
   */
  async loadAvatar(path: string, config: AvatarConfig): Promise<AvatarInfo> {
    // Utiliser l'extension du config si disponible (pour les blob URLs)
    // Sinon, détecter depuis le path
    let extension = config.fileExtension?.toLowerCase();
    
    if (!extension) {
      extension = path.split('.').pop()?.toLowerCase();
    }

    console.log(`📦 Loading avatar: ${path} (type: ${extension})`);

    switch (extension) {
      case 'vrm':
        return await this.loadVRM(path, config);
      case 'fbx':
        return await this.loadFBX(path, config);
      case 'glb':
      case 'gltf':
        return await this.loadGLTF(path, config);
      default:
        throw new Error(`Unsupported avatar format: ${extension}`);
    }
  }

  /**
   * Charge un avatar VRM
   */
  private async loadVRM(path: string, config: AvatarConfig): Promise<AvatarInfo> {
    const gltf = await this.gltfLoader.loadAsync(path);
    const vrm = gltf.userData['vrm'] as VRM;

    if (!vrm) {
      throw new Error('No VRM data found in file');
    }

    // Rotation pour VRM (regarde vers la caméra)
    VRMUtils.rotateVRM0(vrm);

    // Configuration
    vrm.scene.scale.set(config.scale, config.scale, config.scale);
    vrm.scene.position.set(config.position.x, config.position.y, config.position.z);
    vrm.scene.rotation.set(config.rotation.x, config.rotation.y, config.rotation.z);

    // Activer les ombres
    vrm.scene.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });

    // 🎯 NE PAS initialiser la pose - laisser Kalidokit gérer
    // L'initialisation forcée cause des conflits avec le tracking
    
    // 🔍 Log des rotations initiales des bras
    if (vrm.humanoid) {
      const leftArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
      const rightArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');
      console.log('🔍 Initial arm rotations:');
      console.log('   Left:', leftArm ? 
        `x=${leftArm.rotation.x.toFixed(2)}, y=${leftArm.rotation.y.toFixed(2)}, z=${leftArm.rotation.z.toFixed(2)}` : 
        'not found');
      console.log('   Right:', rightArm ? 
        `x=${rightArm.rotation.x.toFixed(2)}, y=${rightArm.rotation.y.toFixed(2)}, z=${rightArm.rotation.z.toFixed(2)}` : 
        'not found');
    }

    // Extraire les bones VRM
    const bones = new Map<string, THREE.Bone>();
    if (vrm.humanoid) {
      // VRM utilise un système de bones humanoid standardisé
      for (const [boneName, node] of Object.entries(vrm.humanoid.humanBones)) {
        if (node && node.node) {
          bones.set(boneName, node.node as any);
        }
      }
    }

    console.log(`✅ VRM avatar loaded: ${bones.size} humanoid bones`);

    return {
      type: 'vrm',
      object: vrm,
      bones,
      isVRM: true
    };
  }

  /**
   * Charge un avatar FBX (Mixamo)
   */
  private async loadFBX(path: string, config: AvatarConfig): Promise<AvatarInfo> {
    console.log('🔄 Loading FBX file...');
    const fbx = await this.fbxLoader.loadAsync(path);
    console.log('✅ FBX file loaded');

    // Log de la hiérarchie
    console.log('📊 FBX object info:', {
      type: fbx.type,
      name: fbx.name,
      children: fbx.children.length,
      hasGeometry: fbx.children.some(c => (c as THREE.Mesh).geometry !== undefined)
    });

    // Configuration
    console.log('📐 Applying scale:', config.scale);
    fbx.scale.set(config.scale, config.scale, config.scale);
    fbx.position.set(config.position.x, config.position.y, config.position.z);
    
    // IMPORTANT: Mixamo FBX utilise Z-up, Three.js utilise Y-up
    // Rotation de -90° (ou -π/2) sur l'axe X pour corriger l'orientation
    fbx.rotation.x = -Math.PI / 2;
    fbx.rotation.y = config.rotation.y;
    fbx.rotation.z = config.rotation.z;
    
    console.log('🔄 Applied -90° X rotation (Mixamo Z-up → Three.js Y-up)');

    // S'assurer que l'objet est visible
    fbx.visible = true;

    // Compter les meshes
    let meshCount = 0;
    let materialCount = 0;

    // Activer les ombres et logs
    fbx.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        meshCount++;
        const mesh = child as THREE.Mesh;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.visible = true;
        
        if (mesh.material) {
          materialCount++;
          // S'assurer que les matériaux sont visibles
          if (Array.isArray(mesh.material)) {
            mesh.material.forEach(mat => {
              mat.visible = true;
              mat.needsUpdate = true;
            });
          } else {
            mesh.material.visible = true;
            mesh.material.needsUpdate = true;
          }
        }
      }
    });

    console.log('📊 FBX meshes:', meshCount, 'materials:', materialCount);

    // Arrêter et supprimer toutes les animations (important pour les avatars téléchargés avec animation)
    if (fbx.animations && fbx.animations.length > 0) {
      console.log(`⚠️ Found ${fbx.animations.length} animations - Removing them for motion tracking`);
      fbx.animations = [];
    }

    // Réinitialiser toutes les rotations des bones à leur état par défaut
    fbx.traverse((child) => {
      if ((child as any).isBone) {
        const bone = child as THREE.Bone;
        // Réinitialiser la rotation à l'identité (T-pose/A-pose)
        bone.rotation.set(0, 0, 0);
        bone.quaternion.identity();
      }
    });

    console.log('✅ All animations removed and bones reset to default pose');

    // Extraire les bones Mixamo
    const bones = this.extractMixamoBones(fbx);

    console.log(`✅ FBX (Mixamo) avatar loaded: ${bones.size} bones`);

    return {
      type: 'mixamo',
      object: fbx,
      bones,
      isVRM: false
    };
  }

  /**
   * Charge un avatar GLTF/GLB
   */
  private async loadGLTF(path: string, config: AvatarConfig): Promise<AvatarInfo> {
    const gltf = await this.gltfLoader.loadAsync(path);
    const model = gltf.scene;

    // Tenter de détecter si c'est un modèle Mixamo
    const bones = this.extractMixamoBones(model);
    const isMixamo = bones.size > 0;

    // Configuration
    model.scale.set(config.scale, config.scale, config.scale);
    model.position.set(config.position.x, config.position.y, config.position.z);
    
    // Si c'est Mixamo GLB, appliquer la rotation Z-up → Y-up
    if (isMixamo) {
      model.rotation.x = -Math.PI / 2;
      model.rotation.y = config.rotation.y;
      model.rotation.z = config.rotation.z;
      console.log('🔄 Applied -90° X rotation (Mixamo Z-up → Three.js Y-up)');
    } else {
      model.rotation.set(config.rotation.x, config.rotation.y, config.rotation.z);
    }

    // Activer les ombres
    model.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        child.castShadow = true;
        child.receiveShadow = true;
      }
    });

    // Arrêter et supprimer toutes les animations
    if (gltf.animations && gltf.animations.length > 0) {
      console.log(`⚠️ Found ${gltf.animations.length} animations - Removing them for motion tracking`);
      gltf.animations = [];
      
      // Réinitialiser toutes les rotations des bones à leur état par défaut
      model.traverse((child) => {
        if ((child as any).isBone) {
          const bone = child as THREE.Bone;
          bone.rotation.set(0, 0, 0);
          bone.quaternion.identity();
        }
      });
      
      console.log('✅ All animations removed and bones reset to default pose');
    } else {
      console.log('✅ No animations found - Avatar is in default pose');
    }

    const avatarType = isMixamo ? 'mixamo' : 'glb';
    console.log(`✅ GLB avatar loaded: ${bones.size} bones (type: ${avatarType})`);

    return {
      type: avatarType as AvatarType,
      object: model,
      bones,
      isVRM: false
    };
  }

  /**
   * Extrait les bones Mixamo d'un modèle
   */
  private extractMixamoBones(object: THREE.Object3D): Map<string, THREE.Bone> {
    const bones = new Map<string, THREE.Bone>();

    object.traverse((child) => {
      if ((child as THREE.Bone).isBone) {
        const bone = child as THREE.Bone;
        
        // Vérifier si c'est un bone Mixamo
        if (bone.name.startsWith('mixamorig')) {
          const mappedName = this.MIXAMO_BONE_MAP[bone.name];
          if (mappedName) {
            bones.set(mappedName, bone);
          }
        } else {
          // Ajouter le bone avec son nom original
          bones.set(bone.name, bone);
        }
      }
    });

    return bones;
  }

  /**
   * 🎯 Initialise la pose du VRM (baisser les bras de la T-pose)
   */
  private initializeVRMPose(vrm: VRM): void {
    if (!vrm.humanoid) return;

    // Baisser les bras à ~45 degrés pour une pose neutre
    const leftUpperArm = vrm.humanoid.getNormalizedBoneNode('leftUpperArm');
    const rightUpperArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');

    if (leftUpperArm) {
      // Rotation Z positive pour baisser le bras gauche
      leftUpperArm.rotation.z = THREE.MathUtils.degToRad(45);
      console.log('🎯 Left arm lowered from T-pose');
    }

    if (rightUpperArm) {
      // Rotation Z négative pour baisser le bras droit
      rightUpperArm.rotation.z = THREE.MathUtils.degToRad(-45);
      console.log('🎯 Right arm lowered from T-pose');
    }

    // Mettre à jour le VRM
    vrm.update(0);
  }

  /**
   * Applique le tracking Kalidokit à un avatar VRM
   */
  applyTrackingToVRM(vrm: VRM, results: KalidoKitResults): void {
    if (!vrm.humanoid) return;

    const riggedResults = results;
    
    // 🎯 Facteur de lissage pour mouvements plus naturels
    const LERP_FACTOR = 0.35; // Plus élevé = plus réactif (0.2-0.5 recommandé)

    // Head rotation
    if (riggedResults.Face?.head?.degrees && vrm.humanoid.getNormalizedBoneNode('head')) {
      const headBone = vrm.humanoid.getNormalizedBoneNode('head')!;
      const targetRotation = new THREE.Euler(
        THREE.MathUtils.degToRad(riggedResults.Face.head.degrees.x),
        THREE.MathUtils.degToRad(riggedResults.Face.head.degrees.y),
        THREE.MathUtils.degToRad(riggedResults.Face.head.degrees.z)
      );
      
      // 🎯 Interpolation pour mouvement fluide
      headBone.rotation.x = THREE.MathUtils.lerp(headBone.rotation.x, targetRotation.x, LERP_FACTOR);
      headBone.rotation.y = THREE.MathUtils.lerp(headBone.rotation.y, targetRotation.y, LERP_FACTOR);
      headBone.rotation.z = THREE.MathUtils.lerp(headBone.rotation.z, targetRotation.z, LERP_FACTOR);
    }

    // Spine rotation
    if (riggedResults.Pose?.Spine && vrm.humanoid.getNormalizedBoneNode('spine')) {
      const spineBone = vrm.humanoid.getNormalizedBoneNode('spine')!;
      // 🎯 Interpolation pour mouvement fluide du torse
      spineBone.rotation.x = THREE.MathUtils.lerp(spineBone.rotation.x, riggedResults.Pose.Spine.x || 0, LERP_FACTOR);
      spineBone.rotation.y = THREE.MathUtils.lerp(spineBone.rotation.y, riggedResults.Pose.Spine.y || 0, LERP_FACTOR);
      spineBone.rotation.z = THREE.MathUtils.lerp(spineBone.rotation.z, riggedResults.Pose.Spine.z || 0, LERP_FACTOR);
    }

    // Hips
    if (riggedResults.Pose?.Hips?.rotation && vrm.humanoid.getNormalizedBoneNode('hips')) {
      const hipsBone = vrm.humanoid.getNormalizedBoneNode('hips')!;
      // 🎯 Interpolation pour mouvement fluide des hanches
      hipsBone.rotation.x = THREE.MathUtils.lerp(hipsBone.rotation.x, riggedResults.Pose.Hips.rotation.x || 0, LERP_FACTOR);
      hipsBone.rotation.y = THREE.MathUtils.lerp(hipsBone.rotation.y, riggedResults.Pose.Hips.rotation.y || 0, LERP_FACTOR);
      hipsBone.rotation.z = THREE.MathUtils.lerp(hipsBone.rotation.z, riggedResults.Pose.Hips.rotation.z || 0, LERP_FACTOR);
    }

    // Left Arm
    this.applyArmRotation(vrm, 'left', riggedResults.Pose);
    // Right Arm
    this.applyArmRotation(vrm, 'right', riggedResults.Pose);

    // 🔍 Debug occasionnel (toutes les 2 secondes environ)
    if (Math.random() < 0.01) {
      const leftArm = vrm.humanoid?.getNormalizedBoneNode('leftUpperArm');
      const rightArm = vrm.humanoid?.getNormalizedBoneNode('rightUpperArm');
      console.log('🎯 TRACKING DEBUG:');
      console.log('   Kalidokit LeftArm:', riggedResults.Pose?.LeftUpperArm);
      console.log('   Kalidokit RightArm:', riggedResults.Pose?.RightUpperArm);
      console.log('   VRM LeftArm rotation:', leftArm ? 
        `x=${leftArm.rotation.x.toFixed(2)}, y=${leftArm.rotation.y.toFixed(2)}, z=${leftArm.rotation.z.toFixed(2)}` : 'N/A');
      console.log('   VRM RightArm rotation:', rightArm ? 
        `x=${rightArm.rotation.x.toFixed(2)}, y=${rightArm.rotation.y.toFixed(2)}, z=${rightArm.rotation.z.toFixed(2)}` : 'N/A');
    }

    // Left Leg
    this.applyLegRotation(vrm, 'left', riggedResults.Pose);
    // Right Leg
    this.applyLegRotation(vrm, 'right', riggedResults.Pose);

    // 🔍 Debug jambes - log occasionnel
    if (Math.random() < 0.01) {
      const leftLeg = vrm.humanoid?.getNormalizedBoneNode('leftUpperLeg');
      const rightLeg = vrm.humanoid?.getNormalizedBoneNode('rightUpperLeg');
      console.log('🦵 LEG TRACKING DEBUG:');
      console.log('   Kalidokit LeftUpperLeg:', riggedResults.Pose?.LeftUpperLeg);
      console.log('   Kalidokit RightUpperLeg:', riggedResults.Pose?.RightUpperLeg);
      console.log('   VRM LeftLeg rotation:', leftLeg ? 
        `x=${leftLeg.rotation.x.toFixed(2)}, y=${leftLeg.rotation.y.toFixed(2)}, z=${leftLeg.rotation.z.toFixed(2)}` : 'N/A');
      console.log('   VRM RightLeg rotation:', rightLeg ? 
        `x=${rightLeg.rotation.x.toFixed(2)}, y=${rightLeg.rotation.y.toFixed(2)}, z=${rightLeg.rotation.z.toFixed(2)}` : 'N/A');
    }

    // Hands
    if (riggedResults.LeftHand) {
      this.applyHandRotation(vrm, 'left', riggedResults.LeftHand);
    }
    if (riggedResults.RightHand) {
      this.applyHandRotation(vrm, 'right', riggedResults.RightHand);
    }

    // Update VRM
    vrm.update(0.016); // ~60fps
  }

  /**
   * Applique le tracking Kalidokit à un avatar Mixamo
   */
  applyTrackingToMixamo(bones: Map<string, THREE.Bone>, results: KalidoKitResults): void {
    // Head rotation
    if (results.Face?.head?.degrees) {
      const headBone = bones.get('Head');
      if (headBone) {
        headBone.rotation.set(
          THREE.MathUtils.degToRad(results.Face.head.degrees.x),
          THREE.MathUtils.degToRad(results.Face.head.degrees.y),
          THREE.MathUtils.degToRad(results.Face.head.degrees.z)
        );
      }
    }

    // Spine
    if (results.Pose?.Spine) {
      const spineBone = bones.get('Spine');
      if (spineBone) {
        spineBone.rotation.set(
          results.Pose.Spine.x || 0,
          results.Pose.Spine.y || 0,
          results.Pose.Spine.z || 0
        );
      }
    }

    // Hips
    if (results.Pose?.Hips?.rotation) {
      const hipsBone = bones.get('Hips');
      if (hipsBone) {
        hipsBone.rotation.set(
          results.Pose.Hips.rotation.x || 0,
          results.Pose.Hips.rotation.y || 0,
          results.Pose.Hips.rotation.z || 0
        );
      }
    }

    // Arms
    this.applyArmRotationToBones(bones, 'Left', results.Pose);
    this.applyArmRotationToBones(bones, 'Right', results.Pose);

    // Legs
    this.applyLegRotationToBones(bones, 'Left', results.Pose);
    this.applyLegRotationToBones(bones, 'Right', results.Pose);

    // Hands
    if (results.LeftHand) {
      this.applyHandRotationToBones(bones, 'Left', results.LeftHand);
    }
    if (results.RightHand) {
      this.applyHandRotationToBones(bones, 'Right', results.RightHand);
    }
  }

  /**
   * Helper: Applique la rotation du bras (VRM)
   */
  private applyArmRotation(vrm: VRM, side: 'left' | 'right', pose: any): void {
    const Side = side === 'left' ? 'Left' : 'Right';
    const sideKey = side === 'left' ? 'left' : 'right';
    const LERP_FACTOR = 0.7; // 🎯 TRÈS réactif pour voir les mouvements
    
    // Upper Arm (Épaule + bras supérieur)
    const upperArm = vrm.humanoid?.getNormalizedBoneNode(`${sideKey}UpperArm`);
    if (pose?.[`${Side}UpperArm`] && upperArm) {
      const armData = pose[`${Side}UpperArm`];
      if (armData && (armData.x !== undefined || armData.y !== undefined || armData.z !== undefined)) {
        // 🎯 Utiliser les valeurs directement SANS offset pour tester
        const targetX = (armData.x || 0);
        const targetY = (armData.y || 0);
        const targetZ = (armData.z || 0);
        
        upperArm.rotation.x = THREE.MathUtils.lerp(upperArm.rotation.x, targetX, LERP_FACTOR);
        upperArm.rotation.y = THREE.MathUtils.lerp(upperArm.rotation.y, targetY, LERP_FACTOR);
        upperArm.rotation.z = THREE.MathUtils.lerp(upperArm.rotation.z, targetZ, LERP_FACTOR);
        
        // 🔍 Log détaillé occasionnel
        if (Math.random() < 0.005 && side === 'left') {
          console.log(`💪 ${Side}UpperArm:`, 
            `Kalido(${targetX.toFixed(2)}, ${targetY.toFixed(2)}, ${targetZ.toFixed(2)}) →`,
            `VRM(${upperArm.rotation.x.toFixed(2)}, ${upperArm.rotation.y.toFixed(2)}, ${upperArm.rotation.z.toFixed(2)})`
          );
        }
      }
    }

    // Lower Arm (Avant-bras)
    const lowerArm = vrm.humanoid?.getNormalizedBoneNode(`${sideKey}LowerArm`);
    if (pose?.[`${Side}LowerArm`] && lowerArm) {
      const armData = pose[`${Side}LowerArm`];
      if (armData && (armData.x !== undefined || armData.y !== undefined || armData.z !== undefined)) {
        lowerArm.rotation.x = THREE.MathUtils.lerp(lowerArm.rotation.x, armData.x || 0, LERP_FACTOR);
        lowerArm.rotation.y = THREE.MathUtils.lerp(lowerArm.rotation.y, armData.y || 0, LERP_FACTOR);
        lowerArm.rotation.z = THREE.MathUtils.lerp(lowerArm.rotation.z, armData.z || 0, LERP_FACTOR);
      }
    }
  }

  /**
   * Helper: Applique la rotation de la jambe (VRM)
   */
  private applyLegRotation(vrm: VRM, side: 'left' | 'right', pose: any): void {
    const Side = side === 'left' ? 'Left' : 'Right';
    const sideKey = side === 'left' ? 'left' : 'right';
    const LERP_FACTOR = 0.5; // 🎯 Plus réactif pour les jambes

    // Upper Leg
    const upperLeg = vrm.humanoid?.getNormalizedBoneNode(`${sideKey}UpperLeg`);
    if (pose?.[`${Side}UpperLeg`] && upperLeg) {
      const legData = pose[`${Side}UpperLeg`];
      // 🎯 Vérifier que les données existent
      if (legData && (legData.x !== undefined || legData.y !== undefined || legData.z !== undefined)) {
        upperLeg.rotation.x = THREE.MathUtils.lerp(upperLeg.rotation.x, legData.x || 0, LERP_FACTOR);
        upperLeg.rotation.y = THREE.MathUtils.lerp(upperLeg.rotation.y, legData.y || 0, LERP_FACTOR);
        upperLeg.rotation.z = THREE.MathUtils.lerp(upperLeg.rotation.z, legData.z || 0, LERP_FACTOR);
      }
    }

    // Lower Leg
    const lowerLeg = vrm.humanoid?.getNormalizedBoneNode(`${sideKey}LowerLeg`);
    if (pose?.[`${Side}LowerLeg`] && lowerLeg) {
      const legData = pose[`${Side}LowerLeg`];
      // 🎯 Vérifier que les données existent
      if (legData && (legData.x !== undefined || legData.y !== undefined || legData.z !== undefined)) {
        lowerLeg.rotation.x = THREE.MathUtils.lerp(lowerLeg.rotation.x, legData.x || 0, LERP_FACTOR);
        lowerLeg.rotation.y = THREE.MathUtils.lerp(lowerLeg.rotation.y, legData.y || 0, LERP_FACTOR);
        lowerLeg.rotation.z = THREE.MathUtils.lerp(lowerLeg.rotation.z, legData.z || 0, LERP_FACTOR);
      }
    }
  }

  /**
   * Helper: Applique la rotation de la main (VRM)
   */
  private applyHandRotation(vrm: VRM, side: 'left' | 'right', hand: any): void {
    const sideKey = side === 'left' ? 'left' : 'right';
    const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
    const joints = ['Proximal', 'Intermediate', 'Distal'];

    for (const finger of fingers) {
      for (const joint of joints) {
        const boneName = `${sideKey}${finger}${joint}`;
        const boneNode = vrm.humanoid?.getNormalizedBoneNode(boneName as any);
        const handData = hand[`${side === 'left' ? 'Left' : 'Right'}${finger}${joint}`];

        if (boneNode && handData) {
          boneNode.rotation.set(
            handData.x || 0,
            handData.y || 0,
            handData.z || 0
          );
        }
      }
    }
  }

  /**
   * Helper: Applique la rotation du bras (Bones classiques)
   */
  private applyArmRotationToBones(bones: Map<string, THREE.Bone>, side: string, pose: any): void {
    const upperArm = bones.get(`${side}UpperArm`);
    const lowerArm = bones.get(`${side}LowerArm`);

    if (pose?.[`${side}UpperArm`] && upperArm) {
      upperArm.rotation.set(
        pose[`${side}UpperArm`].x || 0,
        pose[`${side}UpperArm`].y || 0,
        pose[`${side}UpperArm`].z || 0
      );
    }

    if (pose?.[`${side}LowerArm`] && lowerArm) {
      lowerArm.rotation.set(
        pose[`${side}LowerArm`].x || 0,
        pose[`${side}LowerArm`].y || 0,
        pose[`${side}LowerArm`].z || 0
      );
    }
  }

  /**
   * Helper: Applique la rotation de la jambe (Bones classiques)
   */
  private applyLegRotationToBones(bones: Map<string, THREE.Bone>, side: string, pose: any): void {
    const upperLeg = bones.get(`${side}UpperLeg`);
    const lowerLeg = bones.get(`${side}LowerLeg`);

    if (pose?.[`${side}UpperLeg`] && upperLeg) {
      upperLeg.rotation.set(
        pose[`${side}UpperLeg`].x || 0,
        pose[`${side}UpperLeg`].y || 0,
        pose[`${side}UpperLeg`].z || 0
      );
    }

    if (pose?.[`${side}LowerLeg`] && lowerLeg) {
      lowerLeg.rotation.set(
        pose[`${side}LowerLeg`].x || 0,
        pose[`${side}LowerLeg`].y || 0,
        pose[`${side}LowerLeg`].z || 0
      );
    }
  }

  /**
   * Helper: Applique la rotation de la main (Bones classiques)
   */
  private applyHandRotationToBones(bones: Map<string, THREE.Bone>, side: string, hand: any): void {
    const fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Little'];
    const joints = ['Proximal', 'Intermediate', 'Distal'];

    for (const finger of fingers) {
      for (const joint of joints) {
        const boneName = `${side}${finger}${joint}`;
        const bone = bones.get(boneName);
        const handData = hand[boneName];

        if (bone && handData) {
          bone.rotation.set(
            handData.x || 0,
            handData.y || 0,
            handData.z || 0
          );
        }
      }
    }
  }
}
