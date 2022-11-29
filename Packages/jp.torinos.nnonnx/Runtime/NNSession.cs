using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using Microsoft.Win32.SafeHandles;
using UnityEngine;

namespace NNOnnx
{

/// <summary>
/// Wrapper around Ort::Session and Ort::IOBinding
/// </summary>
public class NNSession : SafeHandleZeroOrMinusOneIsInvalid
{
    public enum ExecutionProvider
    {
        Unknown = -1,
        Cuda = 0,
        TensorRT
    }

    [AOT.MonoPInvokeCallback(typeof(DebugLogDelegate))]
    static void debugLogFunc(string str) => Debug.Log(str);

    [AOT.MonoPInvokeCallback(typeof(DebugLogDelegate))]
    static void debugLogErrorFunc(string str) => Debug.LogError(str);

    public NNSession() : base(true) {}

    bool _disposed = false;

    ~NNSession()
    {
        if(!_disposed)
        {
            Debug.LogWarning("GarbageCollector disposing of NNSession. Please use NNSession.Release() to manually release the session before start releasing buffer resource..");
            _disposed = true;

            ReleaseHandle();
        }
    }

    static NNSession()
        => NativeMethods.RegisterDebugFunc(debugLogFunc, debugLogErrorFunc);

    protected override bool ReleaseHandle()
    {
        NativeMethods.Dispose(this, !_disposed);
        _disposed = true;

        return true;
    }

    /// <summary>
    /// Release GPU resources and sessions registered with Cuda
    /// This function must be call before start releasing GraphicsBuffer resources
    /// </summary>
    public void Release()
    {
        if(!_disposed) ReleaseHandle();
    }

    /// <summary>
    /// Create an onnx session with Cuda as the execution provider
    /// </summary>
    /// <param name="modelPath">Full path to onnx model</param>
    /// <returns></returns>
    public static NNSession CreateSessionWithCuda(string modelPath)
    {
        if(!File.Exists(modelPath))
        {
            Debug.LogError("Model is Not Found");
            return null;
        }

        var session = NativeMethods.CreateSession(modelPath, ExecutionProvider.Cuda, "");

        if(session.handle == IntPtr.Zero)
        {
            Debug.LogError("Create Session Failed");
            return null;
        }

        session.BuildInfoNodes();
        session.Provider = ExecutionProvider.Cuda;

        return session;
    }

    /// <summary>
    /// Create an onnx session with TesnorRT as the execution provider
    /// If the cache cannot be loaded at the cache path,
    /// optimizing onnx model and converting it to TensorRT Engine is running
    /// when the first NNSession.Inference(). This is often a huge expensive process
    /// </summary>
    /// <param name="modelPath">Full path to onnx model</param>
    /// <param name="cachePath">Full path to engine cache directory</param>
    /// <returns></returns>
    public static NNSession CreateSessionWithTensorRT(string modelPath, string cachePath)
    {
        if(!File.Exists(modelPath))
        {
            Debug.LogError("Model is Not Found");
            return null;
        }

        if(!Directory.Exists(cachePath))
        {
            Debug.LogError("Cache Directory is Not Found");
            return null;
        }

        var session = NativeMethods.CreateSession(modelPath, ExecutionProvider.TensorRT, cachePath);

        if(session.handle == IntPtr.Zero)
        {
            Debug.LogError("Create Session Failed");
            return null;
        }

        session.BuildInfoNodes();
        session.Provider = ExecutionProvider.TensorRT;

        return session;
    }

    /// <summary>
    /// Binds Tensor data on GraphicsBuffer to the specified input node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="buffer"></param>
    public void BindInput(NodeData node, GraphicsBuffer buffer)
    {
        if(_inputBuffers.Contains(buffer)) return;

        NativeMethods.BindInput(this, node.Name, buffer.GetNativeBufferPtr());
        _inputBuffers.Add(buffer);
    }

    /// <summary>
    /// Binds Tensor data on GraphicsBuffer to the specified input node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="buffer"></param>
    /// <param name="shape"></param>
    public void BindInput(NodeData node, GraphicsBuffer buffer, params int[] shape)
    {
        if(_inputBuffers.Contains(buffer)) return;
        if(shape.Length != node.Shape.Length)
            Debug.LogError("The number of node shapes and input shapes do not match");

        NativeMethods.BindInputWithShape(this, node.Name, buffer.GetNativeBufferPtr(), shape);
        _inputBuffers.Add(buffer);
    }

    /// <summary>
    /// Binds Tensor data on GraphicsBuffer to the specified output node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="buffer"></param>
    public void BindOutput(NodeData node, GraphicsBuffer buffer)
    {
        if(_outputBuffers.Contains(buffer)) return;

        NativeMethods.BindOutput(this, node.Name, buffer.GetNativeBufferPtr());
        _outputBuffers.Add(buffer);
    }

    /// <summary>
    /// Binds Tensor data on GraphicsBuffer to the specified output node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="buffer"></param>
    /// <param name="shape"></param>
    public void BindOutput(NodeData node, GraphicsBuffer buffer, params int[] shape)
    {
        if(_outputBuffers.Contains(buffer)) return;
        if(shape.Length != node.Shape.Length)
            Debug.LogError("The number of node shapes and input shapes do not match");

        NativeMethods.BindOutputWithShape(this, node.Name, buffer.GetNativeBufferPtr(), shape);
        _outputBuffers.Add(buffer);
    }

    /// <summary>
    /// Binds Tensor data on MemoryAllocation to the specified input node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="allocation"></param>
    public void BindInput(NodeData node, MemoryAllocation allocation)
        => NativeMethods.BindInputAllocation(this, allocation, node.Index);

    /// <summary>
    /// Binds Tensor data on MemoryAllocation to the specified input node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="allocation"></param>
    /// <param name="shape"></param>
    public void BindInput(NodeData node, MemoryAllocation allocation, params int[] shape)
    {
        if(shape.Length != node.Shape.Length)
            Debug.LogError("The number of node shapes and input shapes do not match");

        NativeMethods.BindInputAllocationWithShape(this, allocation, node.Index, shape);
    }

    /// <summary>
    /// Binds Tensor data on MemoryAllocation to the specified output node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="allocation"></param>
    public void BindOutput(NodeData node, MemoryAllocation allocation)
        => NativeMethods.BindOutputAllocation(this, allocation, node.Index);


    /// <summary>
    /// Binds Tensor data on MemoryAllocation to the specified output node.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="allocation"></param>
    /// <param name="shape"></param>
    public void BindOutput(NodeData node, MemoryAllocation allocation, params int[] shape)
    {
        if(shape.Length != node.Shape.Length)
            Debug.LogError("The number of node shapes and input shapes do not match");

        NativeMethods.BindOutputAllocationWithShape(this, allocation, node.Index, shape);
    }

    /// <summary>
    /// Allocates an area of the specified size on the GPU
    /// </summary>
    /// <param name="sizeInBytes">Number of byte units to be allocated on the GPU</param>
    /// <returns></returns>
    public MemoryAllocation AllocateCudaMemory(int sizeInBytes)
    {
        var alloc = NativeMethods.GetAllocation(this, sizeInBytes);
        _allocations.Add(alloc);

        return alloc;
    }

    /// <summary>
    /// Run inference with current bound tensors
    /// </summary>
    public void Inference()
        => NativeMethods.RunSession(this);

    const int NAMEBUILDER_CAPACITIY = 128;
    const int SHAPEARRAY_CAPACITIY = 8;

    void BuildInfoNodes()
    {
        StringBuilder sb = new(NAMEBUILDER_CAPACITIY);

        _inputNodes = new NodeData[InputCount];
        for(int i = 0; i < _inputNodes.Length; i++)
        {
            long[] shae_long = new long[SHAPEARRAY_CAPACITIY];
            NativeMethods.GetInputInfo(this, i, sb, shae_long, out var shapeCount, out var type);

            int[] shape = new int[shapeCount];
            for(int j = 0; j < shapeCount; j++) shape[j] = (int)shae_long[j];

            _inputNodes[i] = new NodeData(i, sb.ToString(), shape, type);
        }

        _outputNodes = new NodeData[OutputCount];
        for(int i = 0; i < _outputNodes.Length; i++)
        {
            long[] shape_long = new long[SHAPEARRAY_CAPACITIY];
            NativeMethods.GetOutputInfo(this, i, sb, shape_long, out var shapeCount, out var type);

            int[] shape = new int[shapeCount];
            for(int j = 0; j < shapeCount; j++) shape[j] = (int)shape_long[j];

            _outputNodes[i] = new NodeData(i, sb.ToString(), shape, type);
        }
    }

    NodeData[] _inputNodes;
    public ReadOnlySpan<NodeData> InputNodes => _inputNodes;

    NodeData[] _outputNodes;
    public ReadOnlySpan<NodeData> OutputNodes => _outputNodes;

    int InputCount => NativeMethods.GetInputCount(this);
    int OutputCount => NativeMethods.GetOutputCount(this);

    public ExecutionProvider Provider { get; private set; }

    List<MemoryAllocation> _allocations = new();

    List<GraphicsBuffer> _inputBuffers = new();
    List<GraphicsBuffer> _outputBuffers = new();
}

}