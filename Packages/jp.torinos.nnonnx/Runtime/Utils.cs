using System;
using System.Linq;
using UnityEngine;

namespace NNOnnx
{

public static class TensorUtil
{
    static ComputeShader NNPreprocess;
    static ComputeShader NNPostprocess;

    static TensorUtil()
    {
        NNPreprocess = Resources.Load<ComputeShader>("NNPreprocess");
        NNPostprocess = Resources.Load<ComputeShader>("NNPostprocess");
    }

    /// <summary>
    /// The data contained in texture is reordered to match the tensor's shape and
    /// copied into linear memory in GraphicsBuffer.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="tex"></param>
    /// <param name="buffer"></param>
    /// <param name="normType"></param>
    public static void TextureToTensorBuffer(NodeData node, Texture tex, GraphicsBuffer buffer, NormalizeType normType = NormalizeType.ZeroToOne)
    {
        if(node.IsDynamicShape)
            throw new InvalidOperationException($"The shape of '{node.Name}' is dynamic shape. Please specify static shape");

        if(node.Shape.Length != 4 || node.Shape[0] != 1)
            throw new InvalidOperationException($"This function does not support shape of '{node.Name}' node");

        if(buffer.count * buffer.stride != node.ElementCount * node.ElementSize)
            throw new InvalidOperationException("Node and Buffer size do not match");

        bool isNCHW = node.Shape[1] < node.Shape[3];
        int type = isNCHW? 0 : 1;

        Vector2 imageSize = isNCHW?
            new Vector2(node.Shape[3], node.Shape[2]) : new Vector2(node.Shape[2], node.Shape[1]);

        NNPreprocess.SetTexture(type, "_InputTexture", tex);
        NNPreprocess.SetVector("_ImageSize", imageSize);
        NNPreprocess.SetInt("_NormType", (int)normType);
        NNPreprocess.SetBuffer(type, "_Tensor", buffer);

        NNPreprocess.Dispatch(type, (int)imageSize.x / 8, (int)imageSize.y / 8, 1);
    }

    /// <summary>
    /// The data contained in texture is reordered to match the tensor's shape and
    /// copied into linear memory in GraphicsBuffer.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="tex"></param>
    /// <param name="buffer"></param>
    /// <param name="normType"></param>
    /// <param name="shape"></param>
    public static void TextureToTensorBuffer(NodeData node, Texture tex, GraphicsBuffer buffer, NormalizeType normType = NormalizeType.ZeroToOne, params int[] shape)
    {
        if(node.Shape.Length != 4)
            throw new InvalidOperationException($"This function does not support shape of '{node.Name}' node");

        if(node.Shape.Length != shape.Length)
            throw new InvalidOperationException($"The number of node shapes and input shapes do not match");

        if(shape[0] != 1)
            throw new InvalidOperationException($"The batch size of the input shape must be 1");

        int count = shape.Aggregate((now, next) => now * next);

        if(buffer.count * buffer.stride != count * node.ElementSize)
            throw new InvalidOperationException("Node and Buffer size do not match");

        bool isNCHW = shape[1] < shape[3];
        int type = isNCHW? 0 : 1;

        Vector2 imageSize = isNCHW?
            new Vector2(shape[3], shape[2]) : new Vector2(shape[2], shape[1]);

        NNPreprocess.SetTexture(type, "_InputTexture", tex);
        NNPreprocess.SetVector("_ImageSize", imageSize);
        NNPreprocess.SetInt("_NormType", (int)normType);
        NNPreprocess.SetBuffer(type, "_Tensor", buffer);

        NNPreprocess.Dispatch(type, (int)imageSize.x / 8, (int)imageSize.y / 8, 1);
    }

    /// <summary>
    /// The data contained in GraphicsBuffer is reordered to match the texture format and
    /// copied into Unity Texture2D.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="buffer"></param>
    /// <param name="tex"></param>
    /// <param name="normType"></param>
    public static void TensorBufferToTexture(NodeData node, GraphicsBuffer buffer, RenderTexture tex, NormalizeType normType = NormalizeType.ZeroToOne)
    {
        if(node.IsDynamicShape)
            throw new InvalidOperationException($"The shape of '{node.Name}' is dynamic shape. Please specify static shape");

        if(node.Shape.Length != 4 || node.Shape[0] != 1)
            throw new InvalidOperationException($"This function does not support shape of '{node.Name}' node");

        if((node.ElementCount) % (tex.width * tex.height) != 0)
            throw new InvalidOperationException("Node and Texture size do not match");

        if(buffer.count * buffer.stride != node.ElementCount * node.ElementSize)
            throw new InvalidOperationException("Node and Buffer size do not match");

        bool isNCHW = node.Shape[1] < node.Shape[3];

        Vector2 imageSize = isNCHW?
            new Vector2(node.Shape[3], node.Shape[2]) : new Vector2(node.Shape[2], node.Shape[1]);

        int numChannels = isNCHW? node.Shape[1] : node.Shape[3];

        if(numChannels != 1 && numChannels != 3)
            throw new InvalidOperationException($"This function does not support shape of '{node.Name}' node");

        int dispatchId = numChannels == 1 ? 0 : (isNCHW? 1 : 2);
        NNPostprocess.SetBuffer(dispatchId, "_Tensor", buffer);
        NNPostprocess.SetVector("_ImageSize", imageSize);
        NNPostprocess.SetInt("_NormType", (int)normType);
        NNPostprocess.SetTexture(dispatchId, numChannels == 1? "_OutputTexture":"_OutputTextureRGB", tex);

        NNPostprocess.Dispatch(dispatchId, (int)imageSize.x / 8, (int)imageSize.y / 8, 1);
    }

    /// <summary>
    /// The data contained in GraphicsBuffer is reordered to match the texture format and
    /// copied into Unity Texture2D.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="buffer"></param>
    /// <param name="tex"></param>
    /// <param name="normType"></param>
    /// <param name="shape"></param>
    public static void TensorBufferToTexture(NodeData node, GraphicsBuffer buffer, RenderTexture tex, NormalizeType normType = NormalizeType.ZeroToOne, params int[] shape)
    {
        if(node.Shape.Length != 4)
            throw new InvalidOperationException($"This function does not support shape of '{node.Name}' node");

        if(node.Shape.Length != shape.Length)
            throw new InvalidOperationException($"The number of node shapes and input shapes do not match");

        if(shape[0] != 1)
            throw new InvalidOperationException($"The batch size of the input shape must be 1");

        int count = shape.Aggregate((now, next) => now * next);

        if(count % (tex.width * tex.height) != 0)
            throw new InvalidOperationException("Node and Texture size do not match");

        if(buffer.count * buffer.stride != count * node.ElementSize)
            throw new InvalidOperationException("Node and Buffer size do not match");

        bool isNCHW = shape[1] < shape[3];

        Vector2 imageSize = isNCHW?
            new Vector2(shape[3], shape[2]) : new Vector2(shape[2], shape[1]);

        int numChannels = isNCHW? shape[1] : shape[3];

        if(numChannels != 1 && numChannels != 3)
            throw new InvalidOperationException($"This function does not support input shape");

        int dispatchId = numChannels == 1 ? 0 : (isNCHW? 1 : 2);
        NNPostprocess.SetBuffer(dispatchId, "_Tensor", buffer);
        NNPostprocess.SetVector("_ImageSize", imageSize);
        NNPostprocess.SetInt("_NormType", (int)normType);
        NNPostprocess.SetTexture(dispatchId, numChannels == 1? "_OutputTexture":"_OutputTextureRGB", tex);

        NNPostprocess.Dispatch(dispatchId, (int)imageSize.x / 8, (int)imageSize.y / 8, 1);
    }
}

}