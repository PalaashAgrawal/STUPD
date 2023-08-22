using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/on")]
public class on : Randomizer
{
    // objects variables initialisation
    public GameObjectParameter prefabs_mat1;
    public GameObjectParameter prefabs_mat2;
    private GameObject currentInstance1;
    private GameObject currentInstance2;
    private Rigidbody rb;
    public Vector3Parameter objectScale;

    // camera variables initialisation
    public Camera mainCamera;
    public Vector3Parameter cameraPosition;
    public Vector3Parameter cameraRot;

    // material variables initialisation
    public Material material1;
    public Material material2;

    protected override void OnIterationStart()
    {   
        // object variables code
        float rotationY = Random.Range(0,360);
        float ObjectScale1 = Random.Range(1,1.5f); // currentInstance1, smaller scale, 1-1.5
        currentInstance1=GameObject.Instantiate(prefabs_mat1.Sample());
        currentInstance1.transform.position = new Vector3(0,2,0);
        currentInstance1.transform.rotation = Quaternion.Euler(0,rotationY,0);
        currentInstance1.transform.localScale = Vector3.one * ObjectScale1;
        rb = currentInstance1.AddComponent<Rigidbody>();

        currentInstance2=GameObject.Instantiate(prefabs_mat2.Sample());
        currentInstance2.transform.position = new Vector3(0,0,0); 
        currentInstance2.transform.localScale = objectScale.Sample();
        

        // camera randomisation code
        mainCamera.transform.position = cameraPosition.Sample();
        mainCamera.transform.rotation = Quaternion.Euler(cameraRot.Sample());

        // material randomisation code
        material1.color = Random.ColorHSV();
        material2.color = Random.ColorHSV();
    }

    protected override void OnIterationEnd()
    {
        GameObject.Destroy(currentInstance1);
        GameObject.Destroy(currentInstance2);
    }
}