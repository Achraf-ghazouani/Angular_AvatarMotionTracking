/**
 * Configuration pour le système de tracking et l'avatar
 */

export interface MediaPipeConfig {
  modelComplexity: 0 | 1 | 2;
  smoothLandmarks: boolean;
  enableSegmentation: boolean;
  smoothSegmentation: boolean;
  minDetectionConfidence: number;
  minTrackingConfidence: number;
}

export interface ThreeJsConfig {
  antialias: boolean;
  alpha: boolean;
  powerPreference: 'high-performance' | 'low-power' | 'default';
  preserveDrawingBuffer: boolean;
}

export interface AvatarConfig {
  modelPath: string;
  scale: number;
  position: { x: number; y: number; z: number };
  rotation: { x: number; y: number; z: number };
  fileExtension?: string; // Extension pour les fichiers uploadés (blob URLs)
}

export interface AIConfig {
  enabled: boolean;
  modelPath?: string;
  inferenceType: 'onnx' | 'torchscript' | 'none';
  smoothingFactor: number;
  predictionSteps: number;
}

export interface AppConfig {
  mediapipe: MediaPipeConfig;
  threejs: ThreeJsConfig;
  avatar: AvatarConfig;
  ai: AIConfig;
  targetFPS: number;
  maxLatencyMs: number;
}

export const DEFAULT_CONFIG: AppConfig = {
  mediapipe: {
    modelComplexity: 1,
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  },
  threejs: {
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
    preserveDrawingBuffer: false
  },
  avatar: {
    modelPath: 'assets/models/avatar.glb',
    scale: 1,
    position: { x: 0, y: -1, z: 0 },
    rotation: { x: 0, y: 0, z: 0 }
  },
  ai: {
    enabled: false, // Désactivé par défaut jusqu'à implémentation
    modelPath: 'assets/models/motion_correction.onnx',
    inferenceType: 'none',
    smoothingFactor: 0.7,
    predictionSteps: 3
  },
  targetFPS: 30,
  maxLatencyMs: 100
};
