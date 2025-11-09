# Configuration Examples

This file contains various configuration examples for different use cases.

## 🎯 Performance Profiles

### High Performance (Recommended for powerful PCs)

```typescript
// src/app/models/config.model.ts
export const HIGH_PERFORMANCE_CONFIG: AppConfig = {
  mediapipe: {
    modelComplexity: 2,              // Maximum accuracy
    smoothLandmarks: true,
    enableSegmentation: false,        // Disable if not needed
    smoothSegmentation: false,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
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
    enabled: true,
    modelPath: 'assets/models/motion_correction.onnx',
    inferenceType: 'onnx',
    smoothingFactor: 0.7,
    predictionSteps: 5
  },
  targetFPS: 60,
  maxLatencyMs: 50
};
```

### Balanced (Default - Good for most systems)

```typescript
export const BALANCED_CONFIG: AppConfig = {
  mediapipe: {
    modelComplexity: 1,              // Balanced
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
    enabled: false,                  // Disabled for performance
    inferenceType: 'none',
    smoothingFactor: 0.7,
    predictionSteps: 3
  },
  targetFPS: 30,
  maxLatencyMs: 100
};
```

### Low End (For older/slower systems)

```typescript
export const LOW_END_CONFIG: AppConfig = {
  mediapipe: {
    modelComplexity: 0,              // Fast but less accurate
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    minDetectionConfidence: 0.3,     // Lower threshold
    minTrackingConfidence: 0.3
  },
  threejs: {
    antialias: false,                // Disable antialiasing
    alpha: false,
    powerPreference: 'low-power',
    preserveDrawingBuffer: false
  },
  avatar: {
    modelPath: 'assets/models/avatar.glb',
    scale: 1,
    position: { x: 0, y: -1, z: 0 },
    rotation: { x: 0, y: 0, z: 0 }
  },
  ai: {
    enabled: false,
    inferenceType: 'none',
    smoothingFactor: 0.5,
    predictionSteps: 1
  },
  targetFPS: 24,
  maxLatencyMs: 150
};
```

## 🎭 Avatar Configurations

### Ready Player Me Avatar

```typescript
avatar: {
  modelPath: 'assets/models/readyplayerme_avatar.glb',
  scale: 1.5,                        // RPM avatars are smaller
  position: { x: 0, y: -1.5, z: 0 },
  rotation: { x: 0, y: 0, z: 0 }
}
```

### Mixamo Character

```typescript
avatar: {
  modelPath: 'assets/models/mixamo_character.fbx',
  scale: 0.01,                       // Mixamo uses different scale
  position: { x: 0, y: 0, z: 0 },
  rotation: { x: 0, y: Math.PI, z: 0 }  // Often needs 180° rotation
}
```

### VRoid Studio Character

```typescript
avatar: {
  modelPath: 'assets/models/vroid_character.vrm',
  scale: 1,
  position: { x: 0, y: -1, z: 0 },
  rotation: { x: 0, y: 0, z: 0 }
}
```

## 🤖 AI Configurations

### With LSTM Model

```typescript
ai: {
  enabled: true,
  modelPath: 'assets/models/motion_lstm.onnx',
  inferenceType: 'onnx',
  smoothingFactor: 0.8,              // High smoothing with AI
  predictionSteps: 5                 // Predict 5 frames ahead
}
```

### With Transformer Model

```typescript
ai: {
  enabled: true,
  modelPath: 'assets/models/motion_transformer.onnx',
  inferenceType: 'onnx',
  smoothingFactor: 0.6,
  predictionSteps: 10                // Transformers can predict further
}
```

### Without AI (Simple Smoothing)

```typescript
ai: {
  enabled: false,
  inferenceType: 'none',
  smoothingFactor: 0.7,              // Still use simple smoothing
  predictionSteps: 0
}
```

## 📹 Camera Configurations

### HD Quality (1080p)

```typescript
// In tracking.service.ts - startTracking method
const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    width: { ideal: 1920 },
    height: { ideal: 1080 },
    frameRate: { ideal: 30 },
    facingMode: 'user'
  }
});
```

### Standard Quality (720p)

```typescript
const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 30 },
    facingMode: 'user'
  }
});
```

### Low Quality (480p) - Better Performance

```typescript
const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    width: { ideal: 640 },
    height: { ideal: 480 },
    frameRate: { ideal: 30 },
    facingMode: 'user'
  }
});
```

### High FPS (60fps) - Smoother tracking

```typescript
const stream = await navigator.mediaDevices.getUserMedia({
  video: {
    width: { ideal: 1280 },
    height: { ideal: 720 },
    frameRate: { ideal: 60 },        // Requires good webcam
    facingMode: 'user'
  }
});
```

## 🎨 Scene Configurations

### Studio Setup (Professional)

```typescript
// In animation.service.ts - setupLights method

// Key light (main)
const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
keyLight.position.set(5, 10, 5);
keyLight.castShadow = true;

// Fill light (soften shadows)
const fillLight = new THREE.DirectionalLight(0x8899ff, 0.5);
fillLight.position.set(-5, 5, -5);

// Rim light (edge lighting)
const rimLight = new THREE.DirectionalLight(0xffffff, 0.8);
rimLight.position.set(0, 5, -10);

// Ambient (overall brightness)
const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
```

### Outdoor Setup

```typescript
// Sun light (bright directional)
const sunLight = new THREE.DirectionalLight(0xffffee, 1.5);
sunLight.position.set(10, 20, 10);

// Sky light (soft ambient)
const skyLight = new THREE.HemisphereLight(0x87ceeb, 0x654321, 0.6);

// No additional lights needed
```

### Dark/Dramatic Setup

```typescript
// Single strong light
const spotLight = new THREE.SpotLight(0xffffff, 2);
spotLight.position.set(0, 10, 0);
spotLight.angle = Math.PI / 4;
spotLight.penumbra = 0.5;

// Very low ambient
const ambientLight = new THREE.AmbientLight(0x404040, 0.1);
```

## 🔧 Smoothing Configurations

### Very Smooth (Less responsive)

```typescript
// In tracking.service.ts
private readonly SMOOTHING_WINDOW = 10;

private generateWeights(length: number): number[] {
  const weights: number[] = [];
  for (let i = 0; i < length; i++) {
    weights.push(Math.pow(1.2, i));   // Gentler curve
  }
  return weights;
}
```

### Responsive (Less smooth)

```typescript
private readonly SMOOTHING_WINDOW = 3;

private generateWeights(length: number): number[] {
  const weights: number[] = [];
  for (let i = 0; i < length; i++) {
    weights.push(Math.pow(2, i));     // Steeper curve (favor recent)
  }
  return weights;
}
```

### Balanced

```typescript
private readonly SMOOTHING_WINDOW = 5;

private generateWeights(length: number): number[] {
  const weights: number[] = [];
  for (let i = 0; i < length; i++) {
    weights.push(Math.pow(1.5, i));   // Default
  }
  return weights;
}
```

## 🎮 Interaction Configurations

### More Sensitive Raycaster

```typescript
// In animation.service.ts - onMouseMove
this.raycaster.setFromCamera(this.mouse, this.camera);
this.raycaster.params.Points!.threshold = 0.1;  // More sensitive
```

### Less Sensitive Raycaster

```typescript
this.raycaster.params.Points!.threshold = 0.5;  // Less sensitive
```

## 💾 Memory Optimization

### Aggressive Cleanup

```typescript
// In tracking.service.ts
private cleanupBuffers(): void {
  // Clear old data every 100 frames
  if (this.frameCount % 100 === 0) {
    this.smoothingBuffer = this.smoothingBuffer.slice(-this.SMOOTHING_WINDOW);
    
    // Force garbage collection (if available)
    if ((window as any).gc) {
      (window as any).gc();
    }
  }
}
```

### Conservative Memory Usage

```typescript
// Limit buffer sizes
private readonly MAX_BUFFER_SIZE = 50;

// Clear old results
if (this.smoothingBuffer.length > this.MAX_BUFFER_SIZE) {
  this.smoothingBuffer = this.smoothingBuffer.slice(-20);
}
```

## 📱 Mobile Configurations (Future)

```typescript
// Detect mobile
const isMobile = /Android|webOS|iPhone|iPad|iPod/i.test(navigator.userAgent);

export const MOBILE_CONFIG: AppConfig = {
  mediapipe: {
    modelComplexity: 0,              // Must be fast on mobile
    smoothLandmarks: true,
    enableSegmentation: false,
    smoothSegmentation: false,
    minDetectionConfidence: 0.3,
    minTrackingConfidence: 0.3
  },
  threejs: {
    antialias: false,
    alpha: false,
    powerPreference: 'low-power',    // Save battery
    preserveDrawingBuffer: false
  },
  avatar: {
    modelPath: 'assets/models/simple_avatar.glb',  // Use simpler model
    scale: 1,
    position: { x: 0, y: -1, z: 0 },
    rotation: { x: 0, y: 0, z: 0 }
  },
  ai: {
    enabled: false,                  // Too heavy for mobile
    inferenceType: 'none',
    smoothingFactor: 0.5,
    predictionSteps: 0
  },
  targetFPS: 20,                     // Lower target
  maxLatencyMs: 200
};
```

---

## 🔄 Switching Configurations Dynamically

```typescript
// In app.component.ts

loadConfig(profile: 'high' | 'balanced' | 'low') {
  switch(profile) {
    case 'high':
      this.config = HIGH_PERFORMANCE_CONFIG;
      break;
    case 'balanced':
      this.config = BALANCED_CONFIG;
      break;
    case 'low':
      this.config = LOW_END_CONFIG;
      break;
  }
  
  // Reinitialize services
  this.reinitializeServices();
}
```

---

Choose the configuration that best matches your hardware and requirements!
