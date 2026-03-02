# Tank Vision AI — C++ Build (ONNX Runtime)

## Gereksinimler

1. **OpenCV 4.x** — `choco install opencv` veya kaynak koddan
2. **ONNX Runtime GPU** — https://github.com/microsoft/onnxruntime/releases
   - Windows: `onnxruntime-win-x64-gpu-<version>.zip` indir
   - Cikar: `C:\onnxruntime\`
3. **CMake 3.18+** — `choco install cmake`
4. **Visual Studio 2022** (C++ destekli)

## Derleme

```bash
cd tank-vision/cpp
mkdir build && cd build

# ONNX Runtime yolunu belirt
cmake .. -DONNXRUNTIME_ROOT=C:/onnxruntime -DOpenCV_DIR=C:/opencv/build

# Derle
cmake --build . --config Release
```

## Kullanim

```bash
# ONNX modeli ile
tank_vision --onnx C:/tv_data/v3/runs/tank_vision_v3m_r5/weights/best.onnx --source 0

# Video dosyasi
tank_vision --onnx best.onnx --source video.mp4 --conf 0.35

# CPU modu
tank_vision --onnx best.onnx --source 0 --cpu

# Video kayit
tank_vision --onnx best.onnx --source 0 --save output.mp4
```

## Notlar
- GPU modunda CUDA 11.x/12.x + cuDNN gerekir
- `--cpu` flagi ile CPU'da da calisir
- ONNX Runtime DLL'leri otomatik olarak build dizinine kopyalanir
