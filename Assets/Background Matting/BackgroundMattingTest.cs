using UnityEngine;
using UnityEngine.UI;
using NNOnnx;

public class BackgroundMattingTest : MonoBehaviour
{
    [SerializeField] RawImage _srcimage;
    [SerializeField] RawImage _image;

    SourceInput _source;

    GraphicsBuffer _inputBuffer;
    GraphicsBuffer _outputBuffer;

    RenderTexture _outputRT;

    NNSession _session;

    NodeData _inputNode;
    NodeData _outputNode;

    void Start()
    {
        _source = this.GetComponent<SourceInput>();

        string modelPath = Application.streamingAssetsPath + "/rvm_mobilenetv3_1088x1920.onnx";
        string cachePath = Application.streamingAssetsPath + "/EngineCache";

        // If the cache cannot be loaded at the cache path,
        // optimizing onnx model and converting it to TensorRT Engine is running
        // when the first NNSession.Inference(). This is often a huge expensive process
        _session = NNSession.CreateSessionWithTensorRT(modelPath, cachePath);

        _inputNode = _session.InputNodes[0];   // src
        _outputNode = _session.OutputNodes[1]; // pha

        Debug.Log($"Robust Video Matting Run with : {_session.Provider} Execution Provider");


        Debug.Log($"Input, Count : {_session.InputNodes.Length}");

        Debug.Log($"Name : {_inputNode.Name}");
        Debug.Log($"Shape : {string.Join(", ", _inputNode.Shape.ToArray())}");
        Debug.Log($"Type : {_inputNode.Type}");


        Debug.Log($"Output, Count : {_session.OutputNodes.Length}");

        Debug.Log($"Name : {_outputNode.Name}");
        Debug.Log($"Shape : {string.Join(", ", _outputNode.Shape.ToArray())}");
        Debug.Log($"Type : {_outputNode.Type}");

        _inputBuffer = _inputNode.CreateTensorBuffer();
        _outputBuffer = _outputNode.CreateTensorBuffer();

        _outputRT = _outputNode.CreateTensorTexture();

        _session.BindInput(_inputNode, _inputBuffer);
        _session.BindOutput(_outputNode, _outputBuffer);

        // Bind recurrent tesnors
        for(int i = 1; i < _session.InputNodes.Length; i++)
        {
            var node = _session.InputNodes[i];
            MemoryAllocation mem = _session.AllocateCudaMemory(node.ElementCount * node.ElementSize);

            _session.BindInput(node, mem);
            _session.BindOutput(_session.OutputNodes[i+1], mem);
        }

        _image.texture = _outputRT;
        _srcimage.texture = _source.SourceTexture;
    }

    void Update()
    {
        TensorUtil.TextureToTensorBuffer(_inputNode, _source.SourceTexture, _inputBuffer);

        _session.Inference();

        TensorUtil.TensorBufferToTexture(_outputNode, _outputBuffer, _outputRT);
    }

    void OnDestroy()
    {
        _session?.Release();
        _inputBuffer?.Release();
        _outputBuffer?.Release();
        _outputRT?.Release();
    }
}
