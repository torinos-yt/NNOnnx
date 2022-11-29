using System;
using System.Text;
using System.Security;
using System.Runtime.InteropServices;

namespace NNOnnx
{

internal delegate void DebugLogDelegate(string str);

[SuppressUnmanagedCodeSecurity]
internal static class NativeMethods
{
    [DllImport("NNOnnx.dll", CharSet = CharSet.Ansi)]
    public static extern NNSession CreateSession(string modelPath, NNSession.ExecutionProvider provider, string cachePath);

    [DllImport("NNOnnx.dll")]
    public static extern int GetInputCount(NNSession session);

    [DllImport("NNOnnx.dll")]
    public static extern int GetOutputCount(NNSession session);

    [DllImport("NNOnnx.dll", CharSet = CharSet.Ansi)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool GetInputInfo(NNSession session, int index, StringBuilder name, long[] shape, out int shapeCount, out ElementType elementType);

    [DllImport("NNOnnx.dll", CharSet = CharSet.Ansi)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool GetOutputInfo(NNSession session, int index, StringBuilder name, long[] shape, out int shapeCount, out ElementType elementType);

    [DllImport("NNOnnx.dll")]
    public static extern int GetExecutionProvider(NNSession session);

    [DllImport("NNOnnx.dll", CharSet = CharSet.Ansi)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool BindInput(NNSession session, string name, IntPtr buffer);

    [DllImport("NNOnnx.dll", CharSet = CharSet.Ansi)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool BindInputWithShape(NNSession session, string name, IntPtr buffer, int[] shape);

    [DllImport("NNOnnx.dll", CharSet = CharSet.Ansi)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool BindOutput(NNSession session, string name, IntPtr buffer);

    [DllImport("NNOnnx.dll", CharSet = CharSet.Ansi)]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool BindOutputWithShape(NNSession session, string name, IntPtr buffer, int[] shape);

    [DllImport("NNOnnx.dll")]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool BindInputAllocation(NNSession session, MemoryAllocation allocation, int index);

    [DllImport("NNOnnx.dll")]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool BindInputAllocationWithShape(NNSession session, MemoryAllocation allocation, int index, int[] shape);

    [DllImport("NNOnnx.dll")]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool BindOutputAllocation(NNSession session, MemoryAllocation allocation, int index);

    [DllImport("NNOnnx.dll")]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool BindOutputAllocationWithShape(NNSession session, MemoryAllocation allocation, int index, int[] shape);

    [DllImport("NNOnnx.dll")]
    [return: MarshalAs(UnmanagedType.U1)]
    public static extern bool RunSession(NNSession session);

    [DllImport("NNOnnx.dll")]
    public static extern int Dispose(NNSession session, [MarshalAs(UnmanagedType.U1)] bool unmap);


    [DllImport("NNOnnx.dll")]
    public static extern MemoryAllocation GetAllocation(NNSession session, int size);

    [DllImport("NNOnnx.dll")]
    public static extern int GetSize(MemoryAllocation allocation);

    [DllImport("NNOnnx.dll")]
    public static extern void FreeAllocation(MemoryAllocation allocation);

    [DllImport("NNOnnx.dll")]
    public static extern void CopyTo(MemoryAllocation allocation, float[] buffer);

    [DllImport("NNOnnx.dll")]
    public static extern void CopyFrom(MemoryAllocation allocation, float[] buffer);

    [DllImport("NNOnnx.dll")]
    public static extern void CopyToBytes(MemoryAllocation allocation, byte[] buffer);

    [DllImport("NNOnnx.dll")]
    public static extern void CopyFromBytes(MemoryAllocation allocation, byte[] buffer);

    [DllImport("NNOnnx.dll")]
    public static extern void RegisterDebugFunc(DebugLogDelegate fp, DebugLogDelegate fper);
}

}