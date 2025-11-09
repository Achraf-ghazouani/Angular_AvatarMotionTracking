# 🚀 Performance Optimization Guide

## Current Optimizations (Applied)

### MediaPipe Settings
- **Model Complexity**: `1` (balanced - was `2`)
  - `0` = Fast but inaccurate (~30+ FPS)
  - `1` = **BEST BALANCE** (~25-30 FPS) ✅
  - `2` = Accurate but slow (~15-20 FPS)

- **Confidence Thresholds**: `0.5` (was `0.7`)
  - Lower = faster processing, may lose tracking easier
  - Higher = slower processing, more stable tracking

### Video Settings
- **Resolution**: `640x480` (was `1280x720`)
  - **50% reduction** in pixels to process
  - **~2x FPS improvement** expected

- **Frame Rate**: `30fps` (was `60fps`)
  - Reduces camera bandwidth
  - More realistic target for web tracking

## Expected Performance

| Configuration | Expected FPS | Quality | Use Case |
|--------------|-------------|---------|----------|
| **Current (Optimized)** | **25-30 FPS** | Good | **Best for most users** ✅ |
| Previous (High Quality) | 15-20 FPS | Excellent | High-end PCs only |
| Ultra Performance | 35-40 FPS | Fair | Low-end PCs |

## If Still Slow

### Option 1: Lower Model Complexity to 0
```typescript
// In config.model.ts
modelComplexity: 0  // Maximum performance
```

### Option 2: Reduce Resolution Further
```typescript
// In tracking.service.ts
width: { ideal: 480 },
height: { ideal: 360 }
```

### Option 3: Disable Smoothing
```typescript
// In config.model.ts
smoothLandmarks: false
```

## Hardware Requirements

### Minimum (15-20 FPS):
- Intel i3 / AMD Ryzen 3
- 4GB RAM
- Integrated GPU

### Recommended (25-30 FPS):
- Intel i5 / AMD Ryzen 5 ✅
- 8GB RAM
- Dedicated GPU

### Optimal (30+ FPS):
- Intel i7 / AMD Ryzen 7
- 16GB RAM
- Modern GPU

## Browser Performance

**Best to Worst:**
1. Chrome/Edge (best WebGL performance)
2. Firefox
3. Safari (slower WebAssembly)

## Troubleshooting Low FPS

1. **Close other tabs/applications**
2. **Use Chrome/Edge browser**
3. **Disable browser extensions**
4. **Check CPU usage** (should be <80%)
5. **Ensure good lighting** (MediaPipe works harder in low light)
6. **Update graphics drivers**

## Real-time Monitoring

Check the Performance panel in your app:
- **FPS**: Should be 25-30+
- **Latency**: Should be <100ms
- **Quality**: Should be 75%+

If Quality is low but FPS is high, improve lighting/camera position.
If FPS is low, reduce model complexity or resolution.
