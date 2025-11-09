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
   * Applique le tracking Kalidokit à un avatar VRM
   */
  applyTrackingToVRM(vrm: VRM, results: KalidoKitResults): void {
    if (!vrm.humanoid) return;

    const riggedResults = results;

    // Head rotation
    if (riggedResults.Face?.head?.degrees && vrm.humanoid.getNormalizedBoneNode('head')) {
      const headBone = vrm.humanoid.getNormalizedBoneNode('head')!;
      headBone.rotation.set(
        THREE.MathUtils.degToRad(riggedResults.Face.head.degrees.x),
        THREE.MathUtils.degToRad(riggedResults.Face.head.degrees.y),
        THREE.MathUtils.degToRad(riggedResults.Face.head.degrees.z)
      );
    }

    // Spine rotation
    if (riggedResults.Pose?.Spine && vrm.humanoid.getNormalizedBoneNode('spine')) {
      const spineBone = vrm.humanoid.getNormalizedBoneNode('spine')!;
      spineBone.rotation.set(
        riggedResults.Pose.Spine.x || 0,
        riggedResults.Pose.Spine.y || 0,
        riggedResults.Pose.Spine.z || 0
      );
    }

    // Hips
    if (riggedResults.Pose?.Hips?.rotation && vrm.humanoid.getNormalizedBoneNode('hips')) {
      const hipsBone = vrm.humanoid.getNormalizedBoneNode('hips')!;
      hipsBone.rotation.set(
        riggedResults.Pose.Hips.rotation.x || 0,
        riggedResults.Pose.Hips.rotation.y || 0,
        riggedResults.Pose.Hips.rotation.z || 0
      );
    }

    // Left Arm
    this.applyArmRotation(vrm, 'left', riggedResults.Pose);
    // Right Arm
    this.applyArmRotation(vrm, 'right', riggedResults.Pose);

    // Left Leg
    this.applyLegRotation(vrm, 'left', riggedResults.Pose);
    // Right Leg
    this.applyLegRotation(vrm, 'right', riggedResults.Pose);

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

    const upperArm = vrm.humanoid?.getNormalizedBoneNode(`${sideKey}UpperArm`);
    const lowerArm = vrm.humanoid?.getNormalizedBoneNode(`${sideKey}LowerArm`);

    if (pose?.[`${Side}UpperArm`] && upperArm) {
      upperArm.rotation.set(
        pose[`${Side}UpperArm`].x || 0,
        pose[`${Side}UpperArm`].y || 0,
        pose[`${Side}UpperArm`].z || 0
      );
    }

    if (pose?.[`${Side}LowerArm`] && lowerArm) {
      lowerArm.rotation.set(
        pose[`${Side}LowerArm`].x || 0,
        pose[`${Side}LowerArm`].y || 0,
        pose[`${Side}LowerArm`].z || 0
      );
    }
  }

  /**
   * Helper: Applique la rotation de la jambe (VRM)
   */
  private applyLegRotation(vrm: VRM, side: 'left' | 'right', pose: any): void {
    const Side = side === 'left' ? 'Left' : 'Right';
    const sideKey = side === 'left' ? 'left' : 'right';

    const upperLeg = vrm.humanoid?.getNormalizedBoneNode(`${sideKey}UpperLeg`);
    const lowerLeg = vrm.humanoid?.getNormalizedBoneNode(`${sideKey}LowerLeg`);

    if (pose?.[`${Side}UpperLeg`] && upperLeg) {
      upperLeg.rotation.set(
        pose[`${Side}UpperLeg`].x || 0,
        pose[`${Side}UpperLeg`].y || 0,
        pose[`${Side}UpperLeg`].z || 0
      );
    }

    if (pose?.[`${Side}LowerLeg`] && lowerLeg) {
      lowerLeg.rotation.set(
        pose[`${Side}LowerLeg`].x || 0,
        pose[`${Side}LowerLeg`].y || 0,
        pose[`${Side}LowerLeg`].z || 0
      );
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
