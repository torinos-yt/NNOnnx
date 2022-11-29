using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(SourceInput))]
sealed class SourceInputEditor : UnityEditor.Editor
{
    SerializedProperty _CameraSource;
    SerializedProperty _TextureSource;
    SerializedProperty _Resolution;
    SerializedProperty _HFlip;

    void OnEnable()
    {
        _CameraSource = serializedObject.FindProperty("_CameraSource");
        _TextureSource = serializedObject.FindProperty("_TextureSource");
        _Resolution = serializedObject.FindProperty("_Resolution");
        _HFlip = serializedObject.FindProperty("_HFlip");
    }

    void OnSelectSource(object name)
    {
        serializedObject.Update();

        var sourceNameProperty = _CameraSource;
        sourceNameProperty.stringValue = (string)name;
        serializedObject.ApplyModifiedProperties();
    }

    void ShowCameraSourceNameDropdown(Rect rect)
    {
        var menu = new GenericMenu();
        if(WebCamTexture.devices.Length > 0)
        {
            foreach (var source in WebCamTexture.devices)
            {
                var name = source.name;
                menu.AddItem(new GUIContent(name), false, OnSelectSource, name);
            }
        }
        else
        {
            menu.AddItem(new GUIContent("No source available"), false, null);
        }

        menu.DropDown(rect);
    }

    void LayoutSourceNameField()
    {
        EditorGUILayout.BeginHorizontal();

        EditorGUILayout.DelayedTextField(_CameraSource);

        var rect = EditorGUILayout.GetControlRect(false, GUILayout.Width(60));
        if (EditorGUI.DropdownButton(rect, new GUIContent("Select"), FocusType.Keyboard))
            ShowCameraSourceNameDropdown(rect);

        EditorGUILayout.EndHorizontal();
        EditorGUILayout.PropertyField(_Resolution);

        EditorGUILayout.ObjectField(_TextureSource);
        EditorGUILayout.Space(5);

        EditorGUILayout.LabelField("WebCam Setting");
        EditorGUI.indentLevel++;
        EditorGUILayout.PropertyField(_HFlip);
        EditorGUI.indentLevel--;
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();
        
        LayoutSourceNameField();

        serializedObject.ApplyModifiedProperties();
    }

}
