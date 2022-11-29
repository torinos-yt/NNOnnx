#include "main.h"
#include "session.h"

extern "C"
{
	UNITY_INTERFACE_EXPORT NNOnnx::NNSession* CreateSession(const char* modelPath, NNOnnx::ExecutionProvider provider, const char* cachePath)
	{
		return NNOnnx::CreateSession(modelPath, provider, cachePath);
	}

	UNITY_INTERFACE_EXPORT int GetInputCount(NNOnnx::NNSession* session)
	{
		return session ? session->GetInputCount() : -1;
	}

	UNITY_INTERFACE_EXPORT int GetOutputCount(NNOnnx::NNSession* session)
	{
		return session ? session->GetOutputCount() : -1;
	}

	UNITY_INTERFACE_EXPORT bool GetInputInfo(NNOnnx::NNSession* session, int index, char* name, int64_t* shape, int* shapeCount, int* elementType)
	{
		return session ? session->GetInputInfo(index, name, shape, shapeCount, elementType) : false;
	}

	UNITY_INTERFACE_EXPORT bool GetOutputInfo(NNOnnx::NNSession* session, int index, char* name, int64_t* shape, int* shapeCount, int* elementType)
	{
		return session ? session->GetOutputInfo(index, name, shape, shapeCount, elementType) : false;
	}

	UNITY_INTERFACE_EXPORT int GetExecutionProvider(NNOnnx::NNSession* session)
	{
		return session ? static_cast<int>(session->GetExecutionProvider()) : -1;
	}

	UNITY_INTERFACE_EXPORT bool BindInput(NNOnnx::NNSession* session, const char* name, void* buffer)
	{
		return session ? session->BindInputBuffer(name, buffer) : false;
	}

	UNITY_INTERFACE_EXPORT bool BindInputWithShape(NNOnnx::NNSession* session, const char* name, void* buffer, int* shape)
	{
		return session ? session->BindInputBufferWithShape(name, buffer, shape) : false;
	}

	UNITY_INTERFACE_EXPORT bool BindOutput(NNOnnx::NNSession* session, const char* name, void* buffer)
	{
		return session ? session->BindOutputBuffer(name, buffer) : false;
	}

	UNITY_INTERFACE_EXPORT bool BindOutputWithShape(NNOnnx::NNSession* session, const char* name, void* buffer, int* shape)
	{
		return session ? session->BindOutputBufferWithShape(name, buffer, shape) : false;
	}

	UNITY_INTERFACE_EXPORT bool BindInputAllocation(NNOnnx::NNSession* session, Ort::MemoryAllocation* allocation, int index)
	{
		return (session && allocation) ? session->BindInputAllocation(index, allocation) : false;
	}

	UNITY_INTERFACE_EXPORT bool BindInputAllocationWithShape(NNOnnx::NNSession* session, Ort::MemoryAllocation* allocation, int index, int* shape)
	{
		return (session && allocation) ? session->BindInputAllocationWithShape(index, allocation, shape) : false;
	}

	UNITY_INTERFACE_EXPORT bool BindOutputAllocation(NNOnnx::NNSession* session, Ort::MemoryAllocation* allocation, int index)
	{
		return (session && allocation) ? session->BindOutputAllocation(index, allocation) : false;
	}

	UNITY_INTERFACE_EXPORT bool BindOutputAllocationWithShape(NNOnnx::NNSession* session, Ort::MemoryAllocation* allocation, int index, int* shape)
	{
		return (session && allocation) ? session->BindOutputAllocationWithShape(index, allocation, shape) : false;
	}

	UNITY_INTERFACE_EXPORT bool RunSession(NNOnnx::NNSession* session)
	{
		return session ? session->Inference() : false;
	}

	UNITY_INTERFACE_EXPORT void* GetAllocation(NNOnnx::NNSession* session, int size)
	{
		return session ? session->GetAllocation(size) : nullptr;
	}

	UNITY_INTERFACE_EXPORT void Dispose(NNOnnx::NNSession* session, bool unmap)
	{
		if(session) session->Dispose(unmap);
	}

	UNITY_INTERFACE_EXPORT int GetSize(Ort::MemoryAllocation* allocation)
	{
		return allocation ? allocation->size() : -1;
	}

	UNITY_INTERFACE_EXPORT void FreeAllocation(Ort::MemoryAllocation* allocation)
	{
		if(allocation) delete allocation;
	}

	UNITY_INTERFACE_EXPORT void CopyTo(Ort::MemoryAllocation* allocation, void* ptr)
	{
		if(allocation)
			cudaMemcpy(ptr, allocation->get(), allocation->size(), cudaMemcpyDeviceToHost);
	}

	UNITY_INTERFACE_EXPORT void CopyFrom(Ort::MemoryAllocation* allocation, void* ptr)
	{
		if(allocation)
			cudaMemcpy(allocation->get(), ptr, allocation->size(), cudaMemcpyHostToDevice);
	}

	UNITY_INTERFACE_EXPORT void CopyToBytes(Ort::MemoryAllocation* allocation, void* ptr)
	{
		if(allocation)
			cudaMemcpy(ptr, allocation->get(), allocation->size(), cudaMemcpyDeviceToHost);
	}

	UNITY_INTERFACE_EXPORT void CopyFromBytes(Ort::MemoryAllocation* allocation, void* ptr)
	{
		if(allocation)
			cudaMemcpy(allocation->get(), ptr, allocation->size(), cudaMemcpyHostToDevice);
	}


	UNITY_INTERFACE_EXPORT void RegisterDebugFunc(DebugFunction fp, DebugFunction fper)
	{
		NNOnnx::RegisterDebugFunction(fp, fper);
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginLoad(IUnityInterfaces * unityInterfaces)
	{
		NNOnnx::SetUnityGraphics(unityInterfaces->Get<IUnityGraphics>());
	}

	UNITY_INTERFACE_EXPORT void UNITY_INTERFACE_API UnityPluginUnload()
	{
		NNOnnx::Dispose();
	}
}