using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/between")]
public class between : Randomizer
{
    // prefabs selection
    public GameObjectParameter prefabs_mat1;
    public GameObjectParameter prefabs_mat2;
    private GameObject currentInstance1;
    private GameObject currentInstance2;
    private GameObject currentInstance3;
    public FloatParameter random_x1;
    public FloatParameter random_z1;
    public FloatParameter random_x2;
    public FloatParameter random_z2;
    // camera selection 
    public Camera mainCamera;
    public Vector3Parameter cameraPosition;
    // materials selection
    public Material material1;
    public Material material2;

    protected override void OnIterationStart()
    {   
        // object variables code
        // random_x = Random.Range(1.5f,5);
        // random_z = Random.Range(-2.5f,2.5f);

        currentInstance1=GameObject.Instantiate(prefabs_mat1.Sample());
        currentInstance2 = GameObject.Instantiate(prefabs_mat2.Sample());
        currentInstance3 = GameObject.Instantiate(currentInstance2);

        currentInstance1.transform.position = new Vector3(random_x1.Sample(),0,random_z1.Sample());
        currentInstance1.transform.rotation = Random.rotation;
        
        currentInstance2.transform.position = new Vector3(random_x2.Sample(),0,random_z2.Sample());
        currentInstance2.transform.rotation = Random.rotation;
        currentInstance3.transform.position = new Vector3(-currentInstance2.transform.position.x,0,currentInstance2.transform.position.z);
        currentInstance3.transform.rotation = Random.rotation;

        // camera randomisation code
        mainCamera.transform.position = cameraPosition.Sample();
        mainCamera.transform.LookAt(new Vector3(0,0,0)); // can set a manual point to look at

        // material randomisation code
        material1.color = Random.ColorHSV();
        material2.color = Random.ColorHSV();
    }

    protected override void OnIterationEnd()
    {
        GameObject.Destroy(currentInstance1);
        GameObject.Destroy(currentInstance2);
        GameObject.Destroy(currentInstance3);
    }

}