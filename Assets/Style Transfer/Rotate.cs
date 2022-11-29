using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rotate : MonoBehaviour
{
    // Update is called once per frame
    void Update()
    {
        this.transform.rotation *= Quaternion.Euler(.1f,.2f,.45f);
    }
}
