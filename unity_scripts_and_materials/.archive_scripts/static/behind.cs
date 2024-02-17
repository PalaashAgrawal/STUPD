using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/behind")]
public class behind : Randomizer
{
    // objects variables initialisation
    public GameObjectParameter prefabs_mat1;
    public GameObjectParameter prefabs_mat2;
    private GameObject currentInstance1;
    private GameObject currentInstance2;
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
        // float rotationX = Random.Range(-15f,15f);
        float rotationY = Random.Range(-45f,45f);
        float rotationZ = Random.Range(-45f,45f);
        float frontObjectScale = Random.Range(1f,2f); // currentInstance1, smaller scale, 1-1.5
        float rearObjectScale = Random.Range(5,6); // currentInstance2, larger scale, 3-4
        // float cameraX = Random.Range(-1.5f,1.5f); // camera X-shift
        // float cameraY = Random.Range(-1.5f,1.5f); // camera Y-shift

        currentInstance1=GameObject.Instantiate(prefabs_mat1.Sample());
        currentInstance1.transform.position = new Vector3(0,0,5);
        currentInstance1.transform.rotation = Quaternion.Euler(90,rotationY,rotationZ);
        currentInstance1.transform.localScale = Vector3.one * rearObjectScale;

        currentInstance2=GameObject.Instantiate(prefabs_mat2.Sample());
        currentInstance2.transform.position = new Vector3(0,0,-4);
        currentInstance2.transform.rotation = Quaternion.Euler(90,rotationY,rotationZ);
        currentInstance2.transform.localScale = Vector3.one * frontObjectScale;

        // camera randomisation code
        // var distance = cameraDistance.Sample();
        // mainCamera.transform.position = new Vector3(0,10,0);
        // mainCamera.transform.rotation = Quaternion.Euler(90,0,0);
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