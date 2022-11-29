#include "main.h"
#include "session.h"

namespace NNOnnx
{
    void SetUnityGraphics(IUnityGraphics* g)
    {
        unityGraphics = g;
    }

    void Log(const char* msg)
    {
        if (DebugFunc == nullptr)
        {
            std::cerr << msg << std::endl;
        }
        else
        {
            DebugFunc(msg);
        }
    }

    void LogError(const char* msg, int line)
    {
        if (DebugErrorFunc == nullptr)
        {
            std::cerr << msg << std::endl;
        }
        else
        {
            std::string str(msg);
            str += " at line: ";
            str += std::to_string(line);
            DebugErrorFunc(str.c_str());
        }
    }

    void RegisterDebugFunction(DebugFunction fp, DebugFunction fper)
    {
        DebugFunc = fp;
        DebugErrorFunc = fper;
    }

    std::string GetProviderString(ExecutionProvider provider)
    {
        switch(provider)
        {
            case ExecutionProvider::Cuda:
                return "_Cuda";
            case ExecutionProvider::TensorRT:
                return "_TensorRT";
        }

        return "Unknown";
    }

    NNSession* CreateSession(const char* modelPath, ExecutionProvider provider, const char* cachePath)
    {
        std::string path(modelPath);
        path += GetProviderString(provider);

        if(SessionMap.size() == 0 ||
            SessionMap.find(path) == SessionMap.end())
        {
            NNSession* session = new NNSession(modelPath, provider, cachePath);
            SessionMap[path] = session;

            return session;
        }
        else
        {
            return SessionMap.find(path)->second;
        }
    }

    void Dispose()
    {
        for(auto& pair : SessionMap) delete pair.second;
        SessionMap.clear();
    }
} // NNOnnx