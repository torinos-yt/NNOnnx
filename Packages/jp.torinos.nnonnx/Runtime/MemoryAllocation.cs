using System;
using Microsoft.Win32.SafeHandles;

namespace NNOnnx
{

/// <summary>
/// Wrapper around Ort::MemoryAllocation
/// Represents the location of linear memory
/// on the GPU allocated by CUDA
/// </summary>
public class MemoryAllocation : SafeHandleZeroOrMinusOneIsInvalid
{
    public MemoryAllocation() : base(true) {}

    protected override bool ReleaseHandle()
    {
        NativeMethods.FreeAllocation(this);
        return true;
    }

    /// <summary>
    /// Copies all values on the GPU array to CPU array
    /// </summary>
    /// <param name="dst">Copy destination array. Array length must match MemoryAllocation.SizeInBytes / sizeof(float)</param>
    public void CopyTo(float[] dst)
    {
        if(dst.Length * sizeof(float) != SizeInBytes)
            throw new InvalidOperationException("The size of the buffer array must match the size of the MemoryAllocation");
        
        NativeMethods.CopyTo(this, dst);
    }

    /// <summary>
    /// Copies all values on the GPU array to CPU array
    /// </summary>
    /// <param name="dst">Copy destination array. Array length must match MemoryAllocation.SizeInBytes</param>
    public void CopyTo(byte[] dst)
    {
        if(dst.Length != SizeInBytes)
            throw new InvalidOperationException("The size of the buffer array must match the size of the MemoryAllocation");
        
        NativeMethods.CopyToBytes(this, dst);
    }

    /// <summary>
    /// Copies all values on the CPU array to GPU array
    /// </summary>
    /// <param name="src">Copy source array. Array length must match MemoryAllocation.SizeInBytes / sizeof(float)</param>
    public void CopyFrom(float[] src)
    {
        if(src.Length * sizeof(float) != SizeInBytes)
            throw new InvalidOperationException("The size of the buffer array must match the size of the MemoryAllocation");
        
        NativeMethods.CopyFrom(this, src);
    }

    /// <summary>
    /// Copies all values on the CPU array to GPU array
    /// </summary>
    /// <param name="src">Copy source array. Array length must match MemoryAllocation.SizeInBytes</param>
    public void CopyFrom(byte[] src)
    {
        if(src.Length != SizeInBytes)
            throw new InvalidOperationException("The size of the buffer array must match the size of the MemoryAllocation");
        
        NativeMethods.CopyFromBytes(this, src);
    }

    public int SizeInBytes => NativeMethods.GetSize(this);
}

}