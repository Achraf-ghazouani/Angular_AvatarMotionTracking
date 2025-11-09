import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';
import { KalidoKitResults } from '@models/tracking.model';
import { AIConfig } from '@models/config.model';

/**
 * Service de correction IA pour améliorer les mouvements
 * Phase 3: Infrastructure pour intégration future du modèle PyTorch (ONNX/TorchScript)
 * 
 * Note: L'IA est actuellement désactivée. Pour l'activer:
 * 1. Entraîner un modèle LSTM avec PyTorch pour prédire/lisser les mouvements
 * 2. Exporter le modèle en format ONNX (.onnx)
 * 3. Placer le fichier dans assets/models/motion_correction.onnx
 * 4. Activer dans la configuration
 */
@Injectable({
  providedIn: 'root'
})
export class AiCorrectionService {
  private session: any = null; // ONNXRuntime InferenceSession
  private isModelLoaded$ = new BehaviorSubject<boolean>(false);
  private isEnabled$ = new BehaviorSubject<boolean>(false);
  private config?: AIConfig;

  // Buffer pour les séquences temporelles (pour LSTM)
  private sequenceBuffer: KalidoKitResults[] = [];
  private readonly SEQUENCE_LENGTH = 10; // Nombre de frames pour la prédiction

  constructor() {}

  /**
   * Initialise le modèle IA
   */
  async initialize(config: AIConfig): Promise<void> {
    this.config = config;

    if (!config.enabled || config.inferenceType === 'none') {
      console.log('ℹ️ AI correction disabled');
      this.isEnabled$.next(false);
      return;
    }

    try {
      if (config.inferenceType === 'onnx' && config.modelPath) {
        await this.loadONNXModel(config.modelPath);
      } else if (config.inferenceType === 'torchscript') {
        console.warn('⚠️ TorchScript not yet supported. Use ONNX format instead.');
      }
    } catch (error) {
      console.error('❌ Error initializing AI model:', error);
      this.isEnabled$.next(false);
    }
  }

  /**
   * Charge un modèle ONNX
   */
  private async loadONNXModel(modelPath: string): Promise<void> {
    try {
      // Dynamically import ONNX Runtime
      const ort = await import('onnxruntime-web');
      
      console.log('📦 Loading ONNX model from:', modelPath);
      this.session = await ort.InferenceSession.create(modelPath);
      
      this.isModelLoaded$.next(true);
      this.isEnabled$.next(true);
      
      console.log('✅ ONNX model loaded successfully');
      console.log('📊 Model inputs:', this.session.inputNames);
      console.log('📊 Model outputs:', this.session.outputNames);
    } catch (error) {
      console.error('❌ Failed to load ONNX model:', error);
      throw error;
    }
  }

  /**
   * Applique la correction IA sur les résultats du tracking
   */
  async correct(results: KalidoKitResults): Promise<KalidoKitResults> {
    if (!this.isEnabled$.value || !this.session || !this.config) {
      // Si l'IA est désactivée, appliquer uniquement le smoothing simple
      return this.applySimpleSmoothing(results);
    }

    try {
      // Ajouter au buffer de séquence
      this.sequenceBuffer.push(results);
      
      // Maintenir la taille du buffer
      if (this.sequenceBuffer.length > this.SEQUENCE_LENGTH) {
        this.sequenceBuffer.shift();
      }

      // Si pas assez de frames, retourner les résultats bruts
      if (this.sequenceBuffer.length < this.SEQUENCE_LENGTH) {
        return results;
      }

      // Préparer les données pour l'inférence
      const inputTensor = await this.prepareInputTensor(this.sequenceBuffer);
      
      // Exécuter l'inférence
      const outputs = await this.session.run({ input: inputTensor });
      
      // Décoder les résultats
      let corrected = this.decodeOutput(outputs, results);
      
      return corrected;
    } catch (error) {
      console.warn('⚠️ AI correction failed, using original data:', error);
      return results;
    }
  }

  /**
   * Prépare le tenseur d'entrée pour le modèle ONNX
   * Format attendu: [batch_size, sequence_length, features]
   */
  private async prepareInputTensor(sequence: KalidoKitResults[]): Promise<any> {
    // Extraire les features pertinentes de chaque frame
    const features = sequence.map(frame => this.extractFeatures(frame));
    
    // Convertir en tenseur
    const ort = await import('onnxruntime-web');
    const dims = [1, sequence.length, features[0].length]; // [batch, sequence, features]
    const data = new Float32Array(features.flat());
    
    return new ort.Tensor('float32', data, dims);
  }

  /**
   * Extrait les features numériques des résultats Kalidokit
   * À adapter selon la structure du modèle PyTorch
   */
  private extractFeatures(results: KalidoKitResults): number[] {
    const features: number[] = [];

    // Face features (rotation de la tête)
    if (results.Face?.head?.degrees) {
      features.push(
        results.Face.head.degrees.x || 0,
        results.Face.head.degrees.y || 0,
        results.Face.head.degrees.z || 0
      );
    } else {
      features.push(0, 0, 0);
    }

    // Pose features (position du corps)
    if (results.Pose?.Hips?.rotation) {
      features.push(
        results.Pose.Hips.rotation.x || 0,
        results.Pose.Hips.rotation.y || 0,
        results.Pose.Hips.rotation.z || 0
      );
    } else {
      features.push(0, 0, 0);
    }

    // Bras gauche
    if (results.Pose?.LeftUpperArm) {
      features.push(
        results.Pose.LeftUpperArm.x || 0,
        results.Pose.LeftUpperArm.y || 0,
        results.Pose.LeftUpperArm.z || 0
      );
    } else {
      features.push(0, 0, 0);
    }

    // Bras droit
    if (results.Pose?.RightUpperArm) {
      features.push(
        results.Pose.RightUpperArm.x || 0,
        results.Pose.RightUpperArm.y || 0,
        results.Pose.RightUpperArm.z || 0
      );
    } else {
      features.push(0, 0, 0);
    }

    return features;
  }

  /**
   * Décode la sortie du modèle et l'applique aux résultats
   */
  private decodeOutput(outputs: any, originalResults: KalidoKitResults): KalidoKitResults {
    // Créer une copie des résultats
    let corrected: KalidoKitResults = JSON.parse(JSON.stringify(originalResults));

    try {
      // Récupérer le tenseur de sortie
      const outputData = outputs.output.data;
      
      // Appliquer les corrections selon la structure du modèle
      // Exemple: les 3 premières valeurs sont la rotation de la tête
      if (corrected.Face?.head?.degrees && outputData.length >= 3) {
        corrected.Face.head.degrees.x = outputData[0];
        corrected.Face.head.degrees.y = outputData[1];
        corrected.Face.head.degrees.z = outputData[2];
      }

      // Appliquer le smoothing factor
      if (this.config) {
        corrected = this.blendResults(originalResults, corrected, this.config.smoothingFactor);
      }
    } catch (error) {
      console.warn('⚠️ Error decoding model output:', error);
      return originalResults;
    }

    return corrected;
  }

  /**
   * Mélange les résultats originaux et corrigés avec un facteur de smoothing
   */
  private blendResults(
    original: KalidoKitResults,
    corrected: KalidoKitResults,
    factor: number
  ): KalidoKitResults {
    const blended: KalidoKitResults = JSON.parse(JSON.stringify(original));

    // Blend head rotation
    if (original.Face?.head?.degrees && corrected.Face?.head?.degrees) {
      blended.Face = blended.Face || {};
      blended.Face.head = blended.Face.head || corrected.Face.head;
      blended.Face.head.degrees = {
        x: this.lerp(original.Face.head.degrees.x, corrected.Face.head.degrees.x, factor),
        y: this.lerp(original.Face.head.degrees.y, corrected.Face.head.degrees.y, factor),
        z: this.lerp(original.Face.head.degrees.z, corrected.Face.head.degrees.z, factor)
      };
    }

    return blended;
  }

  /**
   * Interpolation linéaire
   */
  private lerp(a: number, b: number, t: number): number {
    return a + (b - a) * t;
  }

  /**
   * Smoothing simple sans IA (fallback)
   */
  private applySimpleSmoothing(results: KalidoKitResults): KalidoKitResults {
    if (!this.config) return results;

    this.sequenceBuffer.push(results);
    
    if (this.sequenceBuffer.length > 5) {
      this.sequenceBuffer.shift();
    }

    if (this.sequenceBuffer.length < 2) {
      return results;
    }

    // Moyenne des dernières frames
    const smoothed: KalidoKitResults = JSON.parse(JSON.stringify(results));

    // Smooth head rotation
    if (smoothed.Face?.head?.degrees) {
      const headRotations = this.sequenceBuffer
        .map(r => r.Face?.head?.degrees)
        .filter(Boolean);
      
      if (headRotations.length > 0) {
        smoothed.Face.head.degrees = {
          x: this.average(headRotations.map(r => r!.x)),
          y: this.average(headRotations.map(r => r!.y)),
          z: this.average(headRotations.map(r => r!.z))
        };
      }
    }

    return smoothed;
  }

  /**
   * Calcule la moyenne
   */
  private average(values: number[]): number {
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  /**
   * Observables
   */
  getModelLoadedState(): Observable<boolean> {
    return this.isModelLoaded$.asObservable();
  }

  getEnabledState(): Observable<boolean> {
    return this.isEnabled$.asObservable();
  }

  /**
   * Nettoyage
   */
  destroy(): void {
    this.sequenceBuffer = [];
    if (this.session) {
      // ONNX session cleanup si nécessaire
      this.session = null;
    }
    this.isModelLoaded$.next(false);
    this.isEnabled$.next(false);
  }
}

/**
 * Guide pour entraîner un modèle PyTorch:
 * 
 * 1. Collecter des données de tracking (séquences de mouvements)
 * 2. Créer un modèle LSTM en PyTorch:
 * 
 * ```python
 * import torch
 * import torch.nn as nn
 * 
 * class MotionCorrectionLSTM(nn.Module):
 *     def __init__(self, input_size=12, hidden_size=64, num_layers=2):
 *         super().__init__()
 *         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
 *         self.fc = nn.Linear(hidden_size, input_size)
 *     
 *     def forward(self, x):
 *         lstm_out, _ = self.lstm(x)
 *         output = self.fc(lstm_out[:, -1, :])  # Dernière frame
 *         return output
 * 
 * # Entraîner le modèle
 * model = MotionCorrectionLSTM()
 * # ... training loop ...
 * 
 * # Exporter en ONNX
 * dummy_input = torch.randn(1, 10, 12)  # [batch, sequence, features]
 * torch.onnx.export(model, dummy_input, "motion_correction.onnx",
 *                   input_names=['input'], output_names=['output'],
 *                   dynamic_axes={'input': {0: 'batch', 1: 'sequence'}})
 * ```
 * 
 * 3. Placer le fichier .onnx dans assets/models/
 * 4. Activer dans la configuration
 */
