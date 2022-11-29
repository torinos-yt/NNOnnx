using System;
using System.IO;
using System.Text;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using NNOnnx;

public class ResNetTest : MonoBehaviour
{
    [SerializeField] List<Texture> _srcTextures;
    [SerializeField] RawImage _image;
    [SerializeField] Text _result;

    int _srcIndex;

    // Specify Tensor shape of inputs and outputs for dynamic shapes
    int[] _inputShape = new int[] {1, 3, 224, 224};
    int[] _outputShape = new int[] {1, 1000};

    GraphicsBuffer _inputBuffer;

    MemoryAllocation _outputMemory;
    float[] _outputArray;

    int _outputElementCount;

    string[] _classes;

    NNSession _session;

    NodeData _inputNode;
    NodeData _outputNode;

    void Start()
    {
        string modelPath = Application.streamingAssetsPath + "/resnet50.onnx";
        string classPath = Application.streamingAssetsPath + "/imagenet_classes.txt";

        _session = NNSession.CreateSessionWithCuda(modelPath);

        _inputNode = _session.InputNodes[0];
        _outputNode = _session.OutputNodes[0];

        Debug.Log($"ResNet50 Image Classification Run with : {_session.Provider} Execution Provider");

        Debug.Log($"Input, Count : {_session.InputNodes.Length}");

        Debug.Log($"Name : {_inputNode.Name}");
        Debug.Log($"Shape : {string.Join(", ", _inputNode.Shape.ToArray())}");
        Debug.Log($"Dynamic Shape : {_inputNode.IsDynamicShape}");
        Debug.Log($"Type : {_inputNode.Type}");


        Debug.Log($"Output, Count : {_session.OutputNodes.Length}");

        Debug.Log($"Name : {_outputNode.Name}");
        Debug.Log($"Shape : {string.Join(", ", _outputNode.Shape.ToArray())}");
        Debug.Log($"Dynamic Shape : {_outputNode.IsDynamicShape}");
        Debug.Log($"Type : {_outputNode.Type}");

        _outputElementCount = _outputShape.Aggregate((now, next) => now * next);

        _inputBuffer = new(GraphicsBuffer.Target.Structured, _inputShape.Aggregate((now, next) => now * next), sizeof(float));
        _outputMemory = _session.AllocateCudaMemory(_outputElementCount * _outputNode.ElementSize);

        _classes = File.ReadLines(classPath).ToArray();

        _outputArray = new float[_outputElementCount];

        _session.BindInput(_inputNode, _inputBuffer, _inputShape);
        _session.BindOutput(_outputNode, _outputMemory, _outputShape);

        _image.texture = _srcTextures[_srcIndex];
    }

    void Update()
    {
        TensorUtil.TextureToTensorBuffer(_inputNode, _srcTextures[_srcIndex], _inputBuffer, shape: _inputShape);

        _session.Inference();

        _outputMemory.CopyTo(_outputArray);

        SortResult();
    }

    void SortResult()
    {
        var indexArray = Enumerable.Range(0, _outputElementCount).ToArray();
        Array.Sort(_outputArray, indexArray);

        _result.text = "";
        StringBuilder b = new();

        for(int i = 0; i < 8; i++)
            b.AppendLine($"{i+1} : {_classes[indexArray[_outputArray.Length-i-1]]} {_outputArray[_outputArray.Length-i-1].ToString("F4")}");

        _result.text = b.ToString();
    }

    void OnDestroy()
    {
        _session?.Release();
        _inputBuffer?.Release();
    }

    public void SwitchTexture(int dir)
    {
        _srcIndex += dir;
        if(_srcIndex < 0) _srcIndex += _srcTextures.Count;
        _srcIndex = _srcIndex % _srcTextures.Count;

        _image.texture = _srcTextures[_srcIndex];
    }
}
