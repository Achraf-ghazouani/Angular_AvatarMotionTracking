import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TrackingService } from './services/tracking.service';
import { AnimationService } from './services/animation.service';
import { AiCorrectionService } from './services/ai-correction.service';
import { AvatarInfo } from './services/avatar-loader.service';
import { DEFAULT_CONFIG, AvatarConfig } from './models/config.model';
import { TrackingState, PerformanceMetrics } from './models/tracking.model';
import { Subject, takeUntil } from 'rxjs';

/**
 * Composant principal de l'application
 * Intègre tous les modules: tracking, animation, IA
 */
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('sceneContainer') sceneContainer!: ElementRef<HTMLDivElement>;

  title = 'Avatar IA - Motion Tracking';

  // État
  trackingState: TrackingState = {
    isTracking: false,
    hasWebcam: false,
    isModelLoaded: false
  };

  performanceMetrics: PerformanceMetrics = {
    fps: 0,
    latency: 0,
    trackingQuality: 0,
    timestamp: 0
  };

  isSceneInitialized = false;
  isAvatarLoaded = false;
  isAiEnabled = false;
  errorMessage = '';
  showSettings = false;
  currentAvatarInfo?: AvatarInfo;
  currentAvatarScale = 1;

  // Configuration
  config = DEFAULT_CONFIG;

  // Cleanup
  private destroy$ = new Subject<void>();

  constructor(
    private trackingService: TrackingService,
    private animationService: AnimationService,
    private aiService: AiCorrectionService
  ) {}

  async ngOnInit() {
    console.log('🚀 Avatar IA Motion Tracking - Starting...');
    await this.initializeServices();
  }

  /**
   * Initialise tous les services
   */
  private async initializeServices(): Promise<void> {
    try {
      // Phase 1: Initialiser MediaPipe
      console.log('📡 Initializing MediaPipe...');
      await this.trackingService.initialize(this.config.mediapipe);

      // Initialiser Three.js après que la vue soit prête
      setTimeout(async () => {
        if (this.sceneContainer) {
          console.log('🎨 Initializing Three.js...');
          await this.animationService.initialize(
            this.sceneContainer.nativeElement,
            this.config.threejs
          );

          // NE PAS charger d'avatar par défaut
          // L'utilisateur doit uploader son propre avatar
          console.log('👤 Waiting for user to upload an avatar...');

          // Démarrer l'animation (sans avatar)
          this.animationService.startAnimation();

          // S'abonner aux états
          this.subscribeToStates();
        }
      }, 100);

      // Phase 3: Initialiser l'IA (désactivé par défaut)
      if (this.config.ai.enabled) {
        console.log('🤖 Initializing AI correction...');
        await this.aiService.initialize(this.config.ai);
      }

    } catch (error) {
      console.error('❌ Initialization error:', error);
      this.errorMessage = 'Failed to initialize services. Please refresh the page.';
    }
  }

  /**
   * S'abonne aux états des services
   */
  private subscribeToStates(): void {
    // État du tracking
    this.trackingService.getTrackingState()
      .pipe(takeUntil(this.destroy$))
      .subscribe(state => {
        this.trackingState = state;
        if (state.error) {
          this.errorMessage = state.error;
        }
      });

    // Métriques de performance
    this.trackingService.getPerformanceMetrics()
      .pipe(takeUntil(this.destroy$))
      .subscribe(metrics => {
        this.performanceMetrics = metrics;
      });

    // Résultats du tracking
    this.trackingService.getTrackingResults()
      .pipe(takeUntil(this.destroy$))
      .subscribe(async results => {
        if (results) {
          // Phase 3: Appliquer la correction IA si activée
          const correctedResults = await this.aiService.correct(results);
          
          // Appliquer à l'avatar
          this.animationService.applyTrackingToAvatar(correctedResults);
        }
      });

    // État de la scène
    this.animationService.getInitializedState()
      .pipe(takeUntil(this.destroy$))
      .subscribe(initialized => {
        this.isSceneInitialized = initialized;
      });

    // État de l'avatar
    this.animationService.getAvatarLoadedState()
      .pipe(takeUntil(this.destroy$))
      .subscribe(loaded => {
        this.isAvatarLoaded = loaded;
      });

    // État de l'IA
    this.aiService.getEnabledState()
      .pipe(takeUntil(this.destroy$))
      .subscribe(enabled => {
        this.isAiEnabled = enabled;
      });
  }

  /**
   * Démarre le tracking
   */
  async startTracking(): Promise<void> {
    if (!this.videoElement) {
      this.errorMessage = 'Video element not ready';
      return;
    }

    try {
      this.errorMessage = '';
      await this.trackingService.startTracking(this.videoElement.nativeElement);
    } catch (error) {
      console.error('❌ Failed to start tracking:', error);
      this.errorMessage = 'Failed to start tracking. Please check camera permissions.';
    }
  }

  /**
   * Arrête le tracking
   */
  stopTracking(): void {
    this.trackingService.stopTracking();
  }

  /**
   * Toggle des paramètres
   */
  toggleSettings(): void {
    this.showSettings = !this.showSettings;
  }

  /**
   * Met à jour la configuration MediaPipe
   */
  updateMediaPipeConfig(key: string, value: any): void {
    (this.config.mediapipe as any)[key] = value;
    // Réinitialiser avec la nouvelle config
    this.trackingService.destroy();
    this.trackingService.initialize(this.config.mediapipe);
  }

  /**
   * Active/désactive l'IA
   */
  async toggleAI(): Promise<void> {
    this.config.ai.enabled = !this.config.ai.enabled;
    if (this.config.ai.enabled) {
      await this.aiService.initialize(this.config.ai);
    } else {
      this.aiService.destroy();
    }
  }

  /**
   * Gère la sélection de fichier d'avatar
   */
  async onAvatarFileSelected(event: Event): Promise<void> {
    const input = event.target as HTMLInputElement;
    if (!input.files || input.files.length === 0) return;

    const file = input.files[0];
    const fileName = file.name;
    const fileExtension = fileName.split('.').pop()?.toLowerCase();

    // Vérifier l'extension
    const supportedExtensions = ['vrm', 'fbx', 'glb', 'gltf'];
    if (!fileExtension || !supportedExtensions.includes(fileExtension)) {
      this.errorMessage = `Unsupported format. Please use: ${supportedExtensions.join(', ')}`;
      return;
    }

    try {
      this.errorMessage = '';
      console.log(`📁 Loading avatar: ${fileName}`);

      // Créer une URL temporaire pour le fichier
      const fileUrl = URL.createObjectURL(file);

      // Créer la configuration de l'avatar
      const avatarConfig: AvatarConfig = {
        modelPath: fileUrl,
        scale: this.getDefaultScaleForType(fileExtension),
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 },
        fileExtension: fileExtension // Passer l'extension pour les blob URLs
      };

      // Charger l'avatar
      await this.animationService.loadAvatar(avatarConfig);

      // Récupérer les infos de l'avatar
      this.currentAvatarInfo = this.animationService.getCurrentAvatarInfo();
      this.currentAvatarScale = avatarConfig.scale;

      console.log(`✅ Avatar loaded successfully: ${this.currentAvatarInfo?.type}`);
      
      // Réinitialiser l'input
      input.value = '';
    } catch (error) {
      console.error('❌ Failed to load avatar:', error);
      this.errorMessage = `Failed to load avatar: ${error}`;
    }
  }

  /**
   * Charge un avatar prédéfini
   */
  async loadPresetAvatar(preset: string): Promise<void> {
    let avatarPath = '';
    let scale = 1;

    switch (preset) {
      case 'default':
        avatarPath = 'assets/models/avatar.glb';
        scale = 1;
        break;
      case 'mixamo-kaya':
        avatarPath = 'assets/models/kaya.fbx';
        scale = 0.01;
        break;
      case 'vrm':
        avatarPath = 'assets/models/avatar.vrm';
        scale = 1;
        break;
    }

    if (!avatarPath) return;

    try {
      this.errorMessage = '';
      const avatarConfig: AvatarConfig = {
        modelPath: avatarPath,
        scale,
        position: { x: 0, y: 0, z: 0 },
        rotation: { x: 0, y: 0, z: 0 }
      };

      await this.animationService.loadAvatar(avatarConfig);
      this.currentAvatarInfo = this.animationService.getCurrentAvatarInfo();
      this.currentAvatarScale = scale;
    } catch (error) {
      console.error('❌ Failed to load preset avatar:', error);
      this.errorMessage = `Avatar not found. Please add ${avatarPath} to your project.`;
    }
  }

  /**
   * Retourne l'échelle par défaut selon le type d'avatar
   */
  private getDefaultScaleForType(extension: string): number {
    switch (extension) {
      case 'fbx':
        return 0.01; // Mixamo FBX sont souvent 100x trop grands
      case 'vrm':
        return 1; // VRM sont déjà à la bonne échelle
      case 'glb':
      case 'gltf':
        return 1; // GLB varient, mais 1 est un bon départ
      default:
        return 1;
    }
  }

  /**
   * Toggle debug helpers
   */
  toggleDebugHelpers(): void {
    // Vous pouvez ajouter cette méthode pour afficher/masquer les helpers
    console.log('Debug helpers toggled');
  }

  /**
   * Ajuste l'échelle de l'avatar
   */
  adjustAvatarScale(delta: number): void {
    this.currentAvatarScale = Math.max(0.01, this.currentAvatarScale + delta);
    this.animationService.setAvatarScale(this.currentAvatarScale);
    console.log('🔧 Avatar scale adjusted to:', this.currentAvatarScale.toFixed(2));
  }

  /**
   * Recentre l'avatar
   */
  resetAvatarPosition(): void {
    this.animationService.recenterAvatar();
    console.log('🎯 Avatar recentered');
  }

  /**
   * Getters pour le template
   */
  get fpsColor(): string {
    const fps = this.performanceMetrics.fps;
    if (fps >= this.config.targetFPS) return '#4ade80';
    if (fps >= this.config.targetFPS * 0.7) return '#fbbf24';
    return '#f87171';
  }

  get latencyColor(): string {
    const latency = this.performanceMetrics.latency;
    if (latency <= this.config.maxLatencyMs) return '#4ade80';
    if (latency <= this.config.maxLatencyMs * 1.5) return '#fbbf24';
    return '#f87171';
  }

  get qualityColor(): string {
    const quality = this.performanceMetrics.trackingQuality;
    if (quality >= 90) return '#4ade80';
    if (quality >= 70) return '#fbbf24';
    return '#f87171';
  }

  get isSystemReady(): boolean {
    return this.isSceneInitialized && 
           this.isAvatarLoaded && 
           this.trackingState.isModelLoaded;
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
    
    this.trackingService.destroy();
    this.animationService.destroy();
    this.aiService.destroy();
  }
}
