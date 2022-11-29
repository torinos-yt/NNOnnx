#include "session.h"
#include "main.h"

namespace NNOnnx
{
    NNSession::NNSession(const char* modelPath, const ExecutionProvider provider, const char* cachePath)
        : _provider(provider)
    {
        _sessionOptions = std::make_unique<Ort::SessionOptions>();
        _sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        _env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "Default");

        const OrtApi& api = Ort::GetApi();

        if(_provider == ExecutionProvider::TensorRT)
        { // TensorRT Provider
            OrtTensorRTProviderOptionsV2* rtOptions = nullptr;
            api.CreateTensorRTProviderOptions(&rtOptions);

            _rtOptions = std::make_unique<OrtTensorRTProviderOptionsV2*>(rtOptions);

            std::vector<const char*> keys
            {
                "device_id", "trt_fp16_enable", "trt_int8_enable",
                "trt_engine_cache_enable", "trt_engine_cache_path",
                "trt_max_partition_iterations"
            };
            std::vector<const char*> values{ "0", "1", "0", "1", cachePath, "10" };

            api.UpdateTensorRTProviderOptions(*_rtOptions, keys.data(), values.data(), keys.size());
            api.SessionOptionsAppendExecutionProvider_TensorRT_V2(static_cast<OrtSessionOptions*>(*_sessionOptions), *_rtOptions);
        }

        { // CUDA Provider
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = 0;
            cudaOptions.arena_extend_strategy = 0;
            cudaOptions.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
            cudaOptions.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
            cudaOptions.do_copy_in_default_stream = 1;

            api.SessionOptionsAppendExecutionProvider_CUDA(static_cast<OrtSessionOptions*>(*_sessionOptions), &cudaOptions);
        }

        { // read model
            std::ifstream bytesStream(modelPath, std::ios::in | std::ios::binary | std::ios::ate);
            std::streamsize numBytes = bytesStream.tellg();
            bytesStream.seekg(0, std::ios::beg);
            std::vector<char> buffer(numBytes);
            bytesStream.read(buffer.data(), numBytes);

            try
            {
                _session = std::make_unique<Ort::Session>(*_env, buffer.data(), numBytes, *_sessionOptions);
            }
            catch (Ort::Exception& e)
            {
                LogError(e.what(), __LINE__);
            }
        }

        auto cpuMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Allocator cpuAllocator(*_session, cpuMemoryInfo);

        _memoryInfo = std::make_unique<Ort::MemoryInfo>("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
        _allocator = std::make_unique<Ort::Allocator>(*_session, *_memoryInfo);

        _binding = std::make_unique<Ort::IoBinding>(*_session);

        for(int i = 0; i < _session->GetInputCount(); ++i)
        {
            auto name = std::string(_session->GetInputName(i, cpuAllocator));
            auto shaperef = _session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            auto type = _session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();

            std::vector<int64_t> shape;
            std::copy(shaperef.begin(), shaperef.end(), std::back_inserter(shape));

            _inputNodes.emplace_back(i, name, shape, type);
        }

        for(int i = 0; i < _session->GetOutputCount(); ++i)
        {
            auto name = std::string(_session->GetOutputName(i, cpuAllocator));
            auto shaperef = _session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            auto type = _session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetElementType();

            std::vector<int64_t> shape;
            std::copy(shaperef.begin(), shaperef.end(), std::back_inserter(shape));

            _outputNodes.emplace_back(i, name, shape, type);
        }
    }

    NNSession::~NNSession()
    {
        const OrtApi& api = Ort::GetApi();

        if(_rtOptions) api.ReleaseTensorRTProviderOptions(*_rtOptions.release());

        if(_inputResourceMap.size() > 0)
            for(auto pair : _inputResourceMap) delete std::get<1>(pair.second);

        if(_outputResourceMap.size() > 0)
            for(auto pair : _outputResourceMap) delete std::get<1>(pair.second);
    }

    bool NNSession::GetInputInfo(int index, char* name, int64_t* shape, int* shapeCount, int* elementType) const
    {
        if(index >= GetInputCount()) return false;

        const NodeData& node = _inputNodes[index];

        strcpy_s(name, 256, node.GetName().data());

        const auto& nodeShape = node.GetShape();
        memcpy(shape, nodeShape.data(), nodeShape.size() * sizeof(int64_t));

        *shapeCount = nodeShape.size();
        *elementType = int(node.GetElementType());

        return true;
    }

    bool NNSession::GetOutputInfo(int index, char* name, int64_t* shape, int* shapeCount, int* elementType) const
    {
        if(index >= GetOutputCount()) return false;

        const NodeData& node = _outputNodes[index];

        strcpy_s(name, 256, node.GetName().data());

        const auto& nodeShape = node.GetShape();
        memcpy(shape, nodeShape.data(), nodeShape.size() * sizeof(int64_t));

        *shapeCount = nodeShape.size();
        *elementType = int(node.GetElementType());

        return true;
    }

    bool NNSession::Inference()
    {
        _binding->SynchronizeInputs();

        try
        {
            _session->Run(Ort::RunOptions(), *_binding);
        }
        catch (Ort::Exception& e)
        {
            LOGERROR(e.what());
        }

        _binding->SynchronizeOutputs();

        return true;
    }

    bool NNSession::BindInputBuffer(const char* name, void* buffer)
    {
        std::string str(name);

        if(_inputResourceMap.find(str) != _inputResourceMap.end())
            delete std::get<1>(_inputResourceMap.find(str)->second);

        cudaGraphicsResource_t* resource = new cudaGraphicsResource_t;

        CUDA_CHECK(cudaGraphicsD3D11RegisterResource(resource,
            (ID3D11Resource*)buffer, cudaGraphicsRegisterFlagsNone));

        for(auto& node : _inputNodes)
        {
            if(node.GetName() == str)
            {
                _inputResourceMap[str] = std::make_tuple(&node, resource);

                CUDA_CHECK(cudaGraphicsMapResources(1, resource));

                size_t size;
                void* outputPtr;
                CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&outputPtr, &size, *resource));

                auto tensor = Ort::Value::CreateTensor(*_memoryInfo, reinterpret_cast<float*>(outputPtr),
                    node.GetElementCount(), node.GetShape().data(), node.GetShape().size());

                _binding->BindInput(node.GetName().data(), tensor);

                return true;
            }
        }

        return true;
    }

    bool NNSession::BindInputBufferWithShape(const char* name, void* buffer, int* shape)
    {
        std::string str(name);

        if(_inputResourceMap.find(str) != _inputResourceMap.end())
            delete std::get<1>(_inputResourceMap.find(str)->second);

        cudaGraphicsResource_t* resource = new cudaGraphicsResource_t;

        CUDA_CHECK(cudaGraphicsD3D11RegisterResource(resource,
            (ID3D11Resource*)buffer, cudaGraphicsRegisterFlagsNone));

        for(auto& node : _inputNodes)
        {
            if(node.GetName() == str)
            {
                _inputResourceMap[str] = std::make_tuple(&node, resource);

                CUDA_CHECK(cudaGraphicsMapResources(1, resource));

                size_t size;
                void* outputPtr;
                CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&outputPtr, &size, *resource));

                size_t count = 1;
                std::vector<int64_t> shape_64;
                for(int i = 0; i < node.GetShape().size(); i++)
                {
                    count *= *shape;
                    shape_64.push_back((int64_t)*shape++);
                }

                auto tensor = Ort::Value::CreateTensor(*_memoryInfo, reinterpret_cast<float*>(outputPtr),
                    count, shape_64.data(), node.GetShape().size());

                _binding->BindInput(node.GetName().data(), tensor);

                return true;
            }
        }

        return true;
    }

    bool NNSession::BindOutputBuffer(const char* name, void* buffer)
    {
        std::string str(name);

        if(_inputResourceMap.find(str) != _inputResourceMap.end())
            delete std::get<1>(_inputResourceMap.find(str)->second);

        cudaGraphicsResource_t* resource = new cudaGraphicsResource_t;

        CUDA_CHECK(cudaGraphicsD3D11RegisterResource(resource,
            (ID3D11Resource*)buffer, cudaGraphicsRegisterFlagsNone));

        for(auto& node : _outputNodes)
        {
            if(node.GetName() == str)
            {
                _outputResourceMap[str] = std::make_tuple(&node, resource);

                CUDA_CHECK(cudaGraphicsMapResources(1, resource));

                size_t size;
                void* outputPtr;
                CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&outputPtr, &size, *resource));

                auto tensor = Ort::Value::CreateTensor(*_memoryInfo, reinterpret_cast<float*>(outputPtr),
                    node.GetElementCount(), node.GetShape().data(), node.GetShape().size());

                _binding->BindOutput(node.GetName().data(), tensor);

                return true;
            }
        }

        return true;
    }

    bool NNSession::BindOutputBufferWithShape(const char* name, void* buffer, int* shape)
    {
        std::string str(name);

        if(_inputResourceMap.find(str) != _inputResourceMap.end())
            delete std::get<1>(_inputResourceMap.find(str)->second);

        cudaGraphicsResource_t* resource = new cudaGraphicsResource_t;

        CUDA_CHECK(cudaGraphicsD3D11RegisterResource(resource,
            (ID3D11Resource*)buffer, cudaGraphicsRegisterFlagsNone));

        for(auto& node : _outputNodes)
        {
            if(node.GetName() == str)
            {
                _outputResourceMap[str] = std::make_tuple(&node, resource);

                CUDA_CHECK(cudaGraphicsMapResources(1, resource));

                size_t size;
                void* outputPtr;
                CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&outputPtr, &size, *resource));

                size_t count = 1;
                std::vector<int64_t> shape_64;
                for(int i = 0; i < node.GetShape().size(); i++)
                {
                    count *= *shape;
                    shape_64.push_back((int64_t)*shape++);
                }

                auto tensor = Ort::Value::CreateTensor(*_memoryInfo, reinterpret_cast<float*>(outputPtr),
                    count, shape_64.data(), node.GetShape().size());

                _binding->BindOutput(node.GetName().data(), tensor);

                return true;
            }
        }

        return true;
    }

    bool NNSession::BindInputAllocation(int index, Ort::MemoryAllocation* allocation)
    {
        const auto& node = _inputNodes[index];

        auto tensor = Ort::Value::CreateTensor(*_memoryInfo, reinterpret_cast<float*>(allocation->get()),
            node.GetElementCount(), node.GetShape().data(), node.GetShape().size());

        _binding->BindInput(node.GetName().data(), tensor);
        
        return true;
    }

    bool NNSession::BindInputAllocationWithShape(int index, Ort::MemoryAllocation* allocation, int* shape)
    {
        const auto& node = _inputNodes[index];

        size_t count = 1;
        std::vector<int64_t> shape_64;
        for(int i = 0; i < node.GetShape().size(); i++)
        {
            count *= *shape;
            shape_64.push_back((int64_t)*shape++);
        }

        auto tensor = Ort::Value::CreateTensor(*_memoryInfo, reinterpret_cast<float*>(allocation->get()),
            count, shape_64.data(), node.GetShape().size());

        _binding->BindInput(node.GetName().data(), tensor);
        
        return true;
    }

    bool NNSession::BindOutputAllocation(int index, Ort::MemoryAllocation* allocation)
    {
        const auto& node = _outputNodes[index];

        auto tensor = Ort::Value::CreateTensor(*_memoryInfo, reinterpret_cast<float*>(allocation->get()),
            node.GetElementCount(), node.GetShape().data(), node.GetShape().size());

        _binding->BindOutput(node.GetName().data(), tensor);
        
        return true;
    }

    bool NNSession::BindOutputAllocationWithShape(int index, Ort::MemoryAllocation* allocation, int* shape)
    {
        const auto& node = _outputNodes[index];
        
        size_t count = 1;
        std::vector<int64_t> shape_64;
        for(int i = 0; i < node.GetShape().size(); i++)
        {
            count *= *shape;
            shape_64.push_back((int64_t)*shape++);
        }

        auto tensor = Ort::Value::CreateTensor(*_memoryInfo, reinterpret_cast<float*>(allocation->get()),
            count, shape_64.data(), node.GetShape().size());

        _binding->BindOutput(node.GetName().data(), tensor);
        
        return true;
    }

    void* NNSession::GetAllocation(int size)
    {
        const auto& api = Ort::GetApi();

        void* ptr;
        api.AllocatorAlloc(*_allocator, size, &ptr);
        auto* result = new Ort::MemoryAllocation(*_allocator, ptr, size);
        
        return result;
    }

    void NNSession::Dispose(bool unmap)
    {
        if(_inputResourceMap.size() > 0)
            for(auto& pair : _inputResourceMap)
            {
                auto resource = std::get<1>(pair.second);
                if(unmap)
                {
                    cudaGraphicsUnmapResources(1, resource);
                    cudaGraphicsUnregisterResource(*resource);
                }

                delete resource;
            }

        _inputResourceMap.clear();

        if(_outputResourceMap.size() > 0)
            for(auto& pair : _outputResourceMap)
            {
                auto resource = std::get<1>(pair.second);
                if(unmap)
                {
                    cudaGraphicsUnmapResources(1, resource);
                    cudaGraphicsUnregisterResource(*resource);
                }

                delete resource;
            }

        _outputResourceMap.clear();

        _binding->ClearBoundInputs();
        _binding->ClearBoundOutputs();
    }


    NodeData::NodeData(int index, std::string name, std::vector<int64_t> shape, ONNXTensorElementDataType type)
        : _index(index), _name(name), _shape(shape), _elementType(type)
    {
        _elementCount = 1;
        for(auto& n : _shape) _elementCount *= n;
    }
}