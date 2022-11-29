using UnityEngine;
using UnityEngine.UI;
using NNOnnx;

public class StyleTransferTest : MonoBehaviour
{
    [SerializeField] Texture _srcTexture;
    [SerializeField] RawImage _image;

    GraphicsBuffer _inputBuffer;
    GraphicsBuffer _outputBuffer;

    RenderTexture _outputRT;

    NNSession _session;

    NodeData _inputNode;
    NodeData _outputNode;

    void Start()
    {
        string modelPath = Application.streamingAssetsPath + "/mosaic-8.onnx";
        string cachePath = Application.streamingAssetsPath + "/EngineCache";

        _session = NNSession.CreateSessionWithCuda(modelPath);

        _inputNode = _session.InputNodes[0];
        _outputNode = _session.OutputNodes[0];

        Debug.Log($"Fast Neural Style Transfer Run with : {_session.Provider} Execution Provider");


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

        _image.texture = _outputRT;
    }

    void Update()
    {
        TensorUtil.TextureToTensorBuffer(_inputNode, _srcTexture, _inputBuffer, NormalizeType.ZeroTo255);

        _session.Inference();

        TensorUtil.TensorBufferToTexture(_outputNode, _outputBuffer, _outputRT, NormalizeType.ZeroTo255);
    }

    void OnDestroy()
    {
        _session?.Release();
        _inputBuffer?.Release();
        _outputBuffer?.Release();
        _outputRT?.Release();
    }
}
