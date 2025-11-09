// Type definitions for external libraries without types

declare module 'kalidokit' {
  export interface Vector {
    x: number;
    y: number;
    z: number;
  }

  export interface FaceResults {
    head?: {
      x: number;
      y: number;
      z: number;
      width: number;
      height: number;
      position: Vector;
      normalized: Vector;
      degrees: Vector;
    };
    eye?: {
      l: number;
      r: number;
    };
    mouth?: {
      x: number;
      y: number;
      shape: {
        A: number;
        E: number;
        I: number;
        O: number;
        U: number;
      };
    };
    brow?: number;
    pupil?: Vector;
  }

  export interface PoseResults {
    RightUpperArm?: Vector;
    LeftUpperArm?: Vector;
    RightLowerArm?: Vector;
    LeftLowerArm?: Vector;
    LeftUpperLeg?: Vector;
    RightUpperLeg?: Vector;
    RightLowerLeg?: Vector;
    LeftLowerLeg?: Vector;
    LeftHand?: Vector;
    RightHand?: Vector;
    Spine?: Vector;
    Hips?: {
      worldPosition: Vector;
      position: Vector;
      rotation: Vector;
    };
  }

  export interface HandResults {
    [key: string]: Vector;
  }

  export namespace Face {
    export function solve(
      lm: any,
      options?: {
        runtime?: string;
        video?: HTMLVideoElement;
        imageSize?: { width: number; height: number };
        smoothBlink?: boolean;
        blinkSettings?: [number, number];
      }
    ): FaceResults;
  }

  export namespace Pose {
    export function solve(
      lm: any,
      lm3d: any,
      options?: {
        runtime?: string;
        video?: HTMLVideoElement;
        imageSize?: { width: number; height: number };
        enableLegs?: boolean;
      }
    ): PoseResults;
  }

  export namespace Hand {
    export function solve(lm: any, side: 'Left' | 'Right'): HandResults;
  }

  export const Vector: {
    lerp(v1: Vector, v2: Vector, t: number): Vector;
    normalize(v: Vector): Vector;
    distance(v1: Vector, v2: Vector): number;
  };
}

// Augmenter Window pour les variables globales
interface Window {
  Kalidokit: typeof import('kalidokit');
}
