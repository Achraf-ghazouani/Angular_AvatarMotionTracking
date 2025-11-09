import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { Holistic, Results } from '@mediapipe/holistic';
import { Camera } from '@mediapipe/camera_utils';
import * as Kalidokit from 'kalidokit';
import { HolisticResults, KalidoKitResults, TrackingState, PerformanceMetrics } from '@models/tracking.model';
import { MediaPipeConfig } from '@models/config.model';

/**
 * Service de tracking facial et corporel avec MediaPipe et Kalidokit
 * Phase 1 & 2: Prototype et correction des mouvements
 */
@Injectable({
  providedIn: 'root'
})
export class TrackingService {
  private holistic: Holistic | null = null;
  private camera: Camera | null = null;
  private videoElement: HTMLVideoElement | null = null;

  // Observables pour les résultats
  private trackingResults$ = new BehaviorSubject<KalidoKitResults | null>(null);
  private rawResults$ = new BehaviorSubject<HolisticResults | null>(null);
  private trackingState$ = new BehaviorSubject<TrackingState>({
    isTracking: false,
    hasWebcam: false,
    isModelLoaded: false
  });
  private performanceMetrics$ = new BehaviorSubject<PerformanceMetrics>({
    fps: 0,
    latency: 0,
    trackingQuality: 0,
    timestamp: 0
  });

  // Performance tracking
  private frameCount = 0;
  private lastFrameTime = 0;
  private fpsUpdateInterval = 1000; // Update FPS every second
  private lastFpsUpdate = 0;
  private processingStartTime = 0;

  // Smoothing buffers for stabilization (Phase 2)
  private smoothingBuffer: KalidoKitResults[] = [];
  private readonly SMOOTHING_WINDOW = 5;

  constructor() {}

  /**
   * Initialise MediaPipe Holistic avec la configuration
   */
  async initialize(config: MediaPipeConfig): Promise<void> {
    try {
      this.holistic = new Holistic({
        locateFile: (file) => {
          return `assets/mediapipe/holistic/${file}`;
        }
      });

      this.holistic.setOptions({
        modelComplexity: config.modelComplexity,
        smoothLandmarks: config.smoothLandmarks,
        enableSegmentation: config.enableSegmentation,
        smoothSegmentation: config.smoothSegmentation,
        minDetectionConfidence: config.minDetectionConfidence,
        minTrackingConfidence: config.minTrackingConfidence
      });

      this.holistic.onResults((results: Results) => this.onResults(results));

      this.updateState({ isModelLoaded: true });
      console.log('✅ MediaPipe Holistic initialized');
    } catch (error) {
      console.error('❌ Error initializing MediaPipe:', error);
      this.updateState({ error: 'Failed to initialize MediaPipe model' });
      throw error;
    }
  }

  /**
   * Démarre le tracking avec la webcam
   */
  async startTracking(videoElement: HTMLVideoElement): Promise<void> {
    if (!this.holistic) {
      throw new Error('MediaPipe not initialized. Call initialize() first.');
    }

    this.videoElement = videoElement;

    try {
      // Vérifier l'accès à la webcam
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30 }
        }
      });

      this.videoElement.srcObject = stream;
      this.updateState({ hasWebcam: true });

      // Initialiser la caméra MediaPipe
      this.camera = new Camera(this.videoElement, {
        onFrame: async () => {
          if (this.holistic && this.videoElement) {
            this.processingStartTime = performance.now();
            await this.holistic.send({ image: this.videoElement });
          }
        },
        width: 1280,
        height: 720
      });

      await this.camera.start();
      this.updateState({ isTracking: true });
      this.lastFrameTime = performance.now();
      this.lastFpsUpdate = performance.now();
      
      console.log('✅ Tracking started');
    } catch (error) {
      console.error('❌ Error starting webcam:', error);
      this.updateState({ 
        error: 'Failed to access webcam. Please grant camera permissions.',
        hasWebcam: false 
      });
      throw error;
    }
  }

  /**
   * Arrête le tracking
   */
  stopTracking(): void {
    if (this.camera) {
      this.camera.stop();
      this.camera = null;
    }

    if (this.videoElement && this.videoElement.srcObject) {
      const stream = this.videoElement.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      this.videoElement.srcObject = null;
    }

    this.updateState({ isTracking: false, hasWebcam: false });
    this.trackingResults$.next(null);
    this.rawResults$.next(null);
    
    console.log('⏹️ Tracking stopped');
  }

  /**
   * Callback pour les résultats MediaPipe
   */
  private onResults(results: Results): void {
    const currentTime = performance.now();
    
    // Calculer la latence
    const latency = currentTime - this.processingStartTime;

    // Calculer le FPS
    this.frameCount++;
    if (currentTime - this.lastFpsUpdate >= this.fpsUpdateInterval) {
      const fps = Math.round((this.frameCount * 1000) / (currentTime - this.lastFpsUpdate));
      this.frameCount = 0;
      this.lastFpsUpdate = currentTime;

      // Mettre à jour les métriques
      this.updateMetrics(fps, latency);
    }

    // Convertir les résultats bruts
    const holisticResults: HolisticResults = {
      faceLandmarks: results.faceLandmarks,
      poseLandmarks: results.poseLandmarks,
      leftHandLandmarks: results.leftHandLandmarks,
      rightHandLandmarks: results.rightHandLandmarks,
      image: results.image
    };

    this.rawResults$.next(holisticResults);

    // Traiter avec Kalidokit
    if (results.poseLandmarks) {
      const kalidoResults = this.processWithKalidokit(results);
      
      // Phase 2: Appliquer le lissage
      const smoothedResults = this.applySmoothing(kalidoResults);
      
      this.trackingResults$.next(smoothedResults);
    }

    this.lastFrameTime = currentTime;
  }

  /**
   * Traite les résultats avec Kalidokit
   */
  private processWithKalidokit(results: Results): KalidoKitResults {
    const kalidoResults: KalidoKitResults = {};

    try {
      // Face tracking
      if (results.faceLandmarks) {
        kalidoResults.Face = Kalidokit.Face.solve(results.faceLandmarks, {
          runtime: 'mediapipe',
          video: this.videoElement!,
          imageSize: { width: 1280, height: 720 },
          smoothBlink: true,
          blinkSettings: [0.25, 0.75]
        });
      }

      // Pose tracking
      if (results.poseLandmarks) {
        kalidoResults.Pose = Kalidokit.Pose.solve(results.poseLandmarks, results.poseLandmarks, {
          runtime: 'mediapipe',
          video: this.videoElement!,
          imageSize: { width: 1280, height: 720 },
          enableLegs: true
        });
      }

      // Hand tracking
      if (results.rightHandLandmarks) {
        kalidoResults.RightHand = Kalidokit.Hand.solve(results.rightHandLandmarks, 'Right');
      }

      if (results.leftHandLandmarks) {
        kalidoResults.LeftHand = Kalidokit.Hand.solve(results.leftHandLandmarks, 'Left');
      }
    } catch (error) {
      console.warn('⚠️ Kalidokit processing error:', error);
    }

    return kalidoResults;
  }

  /**
   * Phase 2: Applique un lissage pour stabiliser les mouvements
   * Corrige les erreurs de Kalidokit avec un filtre à moyenne mobile
   */
  private applySmoothing(results: KalidoKitResults): KalidoKitResults {
    // Ajouter au buffer
    this.smoothingBuffer.push(results);
    
    // Maintenir la taille du buffer
    if (this.smoothingBuffer.length > this.SMOOTHING_WINDOW) {
      this.smoothingBuffer.shift();
    }

    // Si pas assez de données, retourner les résultats bruts
    if (this.smoothingBuffer.length < 2) {
      return results;
    }

    // Créer une copie profonde pour le lissage
    const smoothed: KalidoKitResults = JSON.parse(JSON.stringify(results));

    // Lisser la rotation de la tête
    if (smoothed.Face?.head?.degrees) {
      smoothed.Face.head.degrees = this.smoothRotation(
        this.smoothingBuffer.map(r => r.Face?.head?.degrees).filter(Boolean) as any[]
      );
    }

    // Lisser les positions du corps
    if (smoothed.Pose?.Hips?.rotation) {
      smoothed.Pose.Hips.rotation = this.smoothRotation(
        this.smoothingBuffer.map(r => r.Pose?.Hips?.rotation).filter(Boolean) as any[]
      );
    }

    return smoothed;
  }

  /**
   * Lisse les rotations avec une moyenne pondérée
   */
  private smoothRotation(rotations: Array<{ x: number; y: number; z: number }>): { x: number; y: number; z: number } {
    if (rotations.length === 0) {
      return { x: 0, y: 0, z: 0 };
    }

    const weights = this.generateWeights(rotations.length);
    let totalWeight = 0;
    const smoothed = { x: 0, y: 0, z: 0 };

    rotations.forEach((rotation, index) => {
      const weight = weights[index];
      smoothed.x += rotation.x * weight;
      smoothed.y += rotation.y * weight;
      smoothed.z += rotation.z * weight;
      totalWeight += weight;
    });

    return {
      x: smoothed.x / totalWeight,
      y: smoothed.y / totalWeight,
      z: smoothed.z / totalWeight
    };
  }

  /**
   * Génère des poids pour le lissage (plus récent = plus de poids)
   */
  private generateWeights(length: number): number[] {
    const weights: number[] = [];
    for (let i = 0; i < length; i++) {
      weights.push(Math.pow(1.5, i)); // Croissance exponentielle
    }
    return weights;
  }

  /**
   * Met à jour les métriques de performance
   */
  private updateMetrics(fps: number, latency: number): void {
    const quality = this.calculateTrackingQuality();
    
    this.performanceMetrics$.next({
      fps,
      latency,
      trackingQuality: quality,
      timestamp: Date.now()
    });
  }

  /**
   * Calcule la qualité du tracking (0-100)
   */
  private calculateTrackingQuality(): number {
    const current = this.trackingResults$.value;
    if (!current) return 0;

    let score = 0;
    let maxScore = 0;

    // Face detection
    if (current.Face) {
      score += 25;
    }
    maxScore += 25;

    // Pose detection
    if (current.Pose) {
      score += 25;
    }
    maxScore += 25;

    // Hand detection
    if (current.RightHand) {
      score += 25;
    }
    maxScore += 25;

    if (current.LeftHand) {
      score += 25;
    }
    maxScore += 25;

    return Math.round((score / maxScore) * 100);
  }

  /**
   * Met à jour l'état du tracking
   */
  private updateState(updates: Partial<TrackingState>): void {
    const current = this.trackingState$.value;
    this.trackingState$.next({ ...current, ...updates });
  }

  /**
   * Observables publics
   */
  getTrackingResults(): Observable<KalidoKitResults | null> {
    return this.trackingResults$.asObservable();
  }

  getRawResults(): Observable<HolisticResults | null> {
    return this.rawResults$.asObservable();
  }

  getTrackingState(): Observable<TrackingState> {
    return this.trackingState$.asObservable();
  }

  getPerformanceMetrics(): Observable<PerformanceMetrics> {
    return this.performanceMetrics$.asObservable();
  }

  /**
   * Nettoyage
   */
  destroy(): void {
    this.stopTracking();
    
    if (this.holistic) {
      this.holistic.close();
      this.holistic = null;
    }

    this.smoothingBuffer = [];
  }
}
