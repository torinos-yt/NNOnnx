# NNOnnx

This is an alternative to [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@1.0/manual/index.html) for even faster machine learning inference on Unity in limited situations using the [onnxruntime](https://onnxruntime.ai/) and CUDA api.

By using [CUDA's Graphics Interoperability](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html) feature, NNOnnx uses resources on the GPU such as GraphicsBuffer and Texture directly as CUDA resources without copying them to the CPU. This is more useful for models that require higher resolution image input.

This provides the inference on more diversified onnx models and faster runtime speeds on PC platforms where CUDA available compared to the Unity Barracuda.

NNOnnx is not intended to be a full wrapper around onnxruntime for Unity. If you want to use the full functionality of onnxruntime, use [Microsoft.ML.OnnxRuntime Nuget](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime/).

System Requirements
=================
- Unity 2020.1 or higher
- Windows: x64, D3D11
- NVIDIA GPU
- Path to DLLs contained in
  - CUDA 11.x
  - cuDNN
  - TensorRT (Optionally, but most faster)

For more information on CUDA, cuDNN, and TensorRT version compatibility, please check [here](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) and [here](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html). The easiest way to get these is to install the latest [Azure Kinect Body Tracking SDK](https://learn.microsoft.com/en-us/azure/kinect-dk/body-sdk-download).

Install
=================
NNOnnx uses the [Scoped registry](https://docs.unity3d.com/Manual/upm-scoped.html) feature of Package Manager for installation. Open the Package Manager page in the Project Settings window and
add the following entry to the Scoped Registries list:

- Name: `torinos`
- URL: `https://registry.npmjs.com`
- Scope: `jp.torinos`

Now you can install the package from `My Registries` page in the Package Manager
window.

Related Project
================
- [Unity-TensorRT](https://github.com/aman-tiwari/Unity-TensorRT)
- [TensorFlow Lite for Unity Samples](https://github.com/asus4/tf-lite-unity-sample)