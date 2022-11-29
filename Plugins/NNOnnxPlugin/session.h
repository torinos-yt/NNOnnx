#pragma once

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include <onnxruntime/core/graph/constants.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnxruntime/core/session/onnxruntime_session_options_config_keys.h>
#include <onnxruntime/core/session/onnxruntime_run_options_config_keys.h>
#include <onnxruntime/core/providers/providers.h>

#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <dxgi1_2.h>

namespace NNOnnx
{
    enum class ExecutionProvider
    {
        Cuda,
        TensorRT
    };

    class NodeData
    {

    public:

        NodeData(int index, std::string name, std::vector<int64_t> shape, ONNXTensorElementDataType type);
        ~NodeData() { _shape.clear(); };

        inline std::string GetName() const { return _name; };
        inline std::vector<int64_t> GetShape() const { return _shape; };
        inline ONNXTensorElementDataType GetElementType() const { return _elementType; };
        inline size_t GetElementCount() const { return _elementCount; };

    private:

        int _index;
        size_t _elementCount;
        std::string _name;
        std::vector<int64_t> _shape;
        ONNXTensorElementDataType _elementType;

    };

    class NNSession
    {

    public:

        NNSession(const char* modelPath, const ExecutionProvider provider, const char* cachePath);
        ~NNSession();

        inline int GetInputCount() const { return _session->GetInputCount(); };
        inline int GetOutputCount() const { return _session->GetOutputCount(); };
        inline ExecutionProvider GetExecutionProvider() const { return _provider; };

        bool GetInputInfo(int index, char* name, int64_t* shape, int* shapeCount, int* elementType) const;
        bool GetOutputInfo(int index, char* name, int64_t* shape, int* shapeCount, int* elementType) const;

        bool Inference();

        bool BindInputBuffer(const char* name, void* buffer);
        bool BindOutputBuffer(const char* name, void* buffer);

        bool BindInputAllocation(int index, Ort::MemoryAllocation* allocation);
        bool BindOutputAllocation(int index, Ort::MemoryAllocation* allocation);

        bool BindInputBufferWithShape(const char* name, void* buffer, int* shape);
        bool BindOutputBufferWithShape(const char* name, void* buffer, int* shape);

        bool BindInputAllocationWithShape(int index, Ort::MemoryAllocation* allocation, int* shape);
        bool BindOutputAllocationWithShape(int index, Ort::MemoryAllocation* allocation, int* shape);

        void* GetAllocation(int size);

        void Dispose(bool unmap);

    private:

        std::unique_ptr<OrtTensorRTProviderOptionsV2*> _rtOptions;

        std::unique_ptr<Ort::SessionOptions> _sessionOptions;

        std::unique_ptr<Ort::Session> _session;
        std::unique_ptr<Ort::Env> _env;

        std::unique_ptr<Ort::MemoryInfo> _memoryInfo;
        std::unique_ptr<Ort::Allocator> _allocator;

        std::unordered_map<std::string, std::tuple<NodeData*, cudaGraphicsResource_t*>> _inputResourceMap;
        std::unordered_map<std::string, std::tuple<NodeData*, cudaGraphicsResource_t*>> _outputResourceMap;

        std::unique_ptr<Ort::IoBinding> _binding;

        std::vector<NodeData> _inputNodes;
        std::vector<NodeData> _outputNodes;

        ExecutionProvider _provider;

    };
}