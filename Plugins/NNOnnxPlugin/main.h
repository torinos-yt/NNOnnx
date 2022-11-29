#pragma once

#include "Unity/IUnityInterface.h"
#include "Unity/IUnityGraphics.h"

#include <string>
#include <unordered_map>

using DebugFunction = void(*)(const char*);

#define LOGERROR(X)                \
		do{                        \
			LogError(X, __LINE__); \
			return false;          \
		} while (0)

#define CUDA_CHECK(X) if(X != cudaSuccess) { LOGERROR(cudaGetErrorString(X)); }

namespace NNOnnx
{
	class NNSession;
	enum class ExecutionProvider;

	void RegisterDebugFunction(DebugFunction fp, DebugFunction fper);

	void SetUnityGraphics(IUnityGraphics* g);

	void Log(const char* msg);

	void LogError(const char* msg, int line);

	std::string GetProviderString(ExecutionProvider provider);

	NNSession* CreateSession(const char* path, ExecutionProvider provider, const char* cachePath);

	void Dispose();
}

namespace
{
	DebugFunction DebugFunc = nullptr;
	DebugFunction DebugErrorFunc = nullptr;

	IUnityGraphics* unityGraphics = nullptr;

	std::unordered_map<std::string, NNOnnx::NNSession*> SessionMap;
}
