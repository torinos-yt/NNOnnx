using System;
using System.Linq;
using UnityEngine;

namespace NNOnnx
{

public readonly struct NodeData
{
    readonly public int Index;
    readonly public string Name;
    readonly public ElementType Type;
    readonly public int ElementCount;
    readonly public bool IsDynamicShape;
    readonly public int ElementSize;

    readonly int[] _shape;
    public ReadOnlySpan<int> Shape => _shape;

    internal NodeData(int index, string name, int[] shape, ElementType type)
    {
        (Index, Name, _shape, Type) = (index, name, shape, type);

        IsDynamicShape = !_shape.All(dim => dim > 0);
        ElementCount = IsDynamicShape? -1 : _shape.Aggregate((now, next) => now * next);
        ElementSize = Type.Size();
    }

    public GraphicsBuffer CreateTensorBuffer()
    {
        if(IsDynamicShape)
            throw new InvalidOperationException($"The shape of '{Name}' is dynamic shape");

        return new GraphicsBuffer(GraphicsBuffer.Target.Structured, ElementCount, ElementSize);
    }

    public RenderTexture CreateTensorTexture()
    {
        if(IsDynamicShape)
            throw new InvalidOperationException($"The shape of '{Name}' is dynamic shape");

        if(Shape.Length != 4 || Shape[0] != 1)
            throw new InvalidOperationException($"This function does not support shape of '{Name}' node");

        bool isNCHW = Shape[1] < Shape[3];
        int numChannels = isNCHW? Shape[1] : Shape[3];
        int width = isNCHW? Shape[3] : Shape[2];
        int height = isNCHW? Shape[2] : Shape[1];

        if(numChannels != 1 && numChannels != 3)
            throw new InvalidOperationException($"This function does not support shape of '{Name}' node");

        var format = numChannels == 1? RenderTextureFormat.R8 : RenderTextureFormat.ARGB32;

        var tex = new RenderTexture(width, height, 0, format);
        tex.enableRandomWrite = true;
        tex.Create();

        return tex;
    }
}

public enum NormalizeType
{
    ZeroToOne,
    MinusOneToOne,
    ZeroTo255
}

public enum ElementType
{
  Undefined = 0,
  Float,      // maps to c type float
  Uint8,      // maps to c type uint8_t
  Int8,       // maps to c type int8_t
  Uint16,     // maps to c type uint16_t
  Int16,      // maps to c type int16_t
  Int32,      // maps to c type int32_t
  Int64,      // maps to c type int64_t
  String,     // maps to c++ type std::string
  Bool,
  Float16,
  Double,     // maps to c type double
  Uint32,     // maps to c type uint32_t
  Uint64,     // maps to c type uint64_t
  Complex64,  // complex with float32 real and imaginary components
  Complex32,  // complex with float64 real and imaginary components
  BFloat16
}

public static class ElementSizeExtension
{
    public static int Size(this ElementType type)
    {
        return type switch
        {
            ElementType.Uint8   or
            ElementType.Int8      => 1,
            ElementType.Uint16  or
            ElementType.Int16   or
            ElementType.Float16 or
            ElementType.BFloat16  => 2,
            ElementType.Float   or
            ElementType.Int32   or
            ElementType.Uint32  or
            ElementType.Complex32 => 4,
            ElementType.Double  or
            ElementType.Uint64  or
            ElementType.Complex64 => 8,
            _ => throw new InvalidOperationException()
        };
    }
}

}