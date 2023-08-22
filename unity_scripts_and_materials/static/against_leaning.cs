using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/against_leaning")]
public class against_leaning : Randomizer
{
    // for this script, because the objects are hardcoded, in the UI, only select the prefabs hardcoded here for prefabs_mat1 and prefabs_mat2
    // objects variables initialisation
    public GameObjectParameter prefabs_mat1;
    public GameObjectParameter prefabs_mat2;
    private GameObject currentInstance1;
    private GameObject currentInstance2;
    // public FloatParameter largeObjectScale1;
    // public FloatParameter smallObjectScale1;
    private Rigidbody rb1;
    private Rigidbody rb2;

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
        currentInstance1=GameObject.Instantiate(prefabs_mat1.Sample());
        //hard code some objects that are required
        if (currentInstance1.name=="Arch(Clone)")
        {
            currentInstance1.transform.position = new Vector3(-1,1,2);
            currentInstance1.transform.rotation = Quaternion.Euler(0,0,0);
            currentInstance1.transform.localScale = new Vector3(1,1,1);
            rb1 = currentInstance1.AddComponent<Rigidbody>();
        }
        if (currentInstance1.name=="Cone(Clone)")
        {
            currentInstance1.transform.position = new Vector3(-1,1.5f,2);
            currentInstance1.transform.rotation = Quaternion.Euler(0,0,0);
            currentInstance1.transform.localScale = new Vector3(3,3,3);
            rb1 = currentInstance1.AddComponent<Rigidbody>();
        }
        if (currentInstance1.name=="Cube(Clone)")
        {
            currentInstance1.transform.position = new Vector3(-1,1,2);
            currentInstance1.transform.rotation = Quaternion.Euler(0,0,0);
            currentInstance1.transform.localScale = new Vector3(2,2,2);
            rb1 = currentInstance1.AddComponent<Rigidbody>();
        }
        if (currentInstance1.name=="Pipe(Clone)")
        {
            currentInstance1.transform.position = new Vector3(-0.75f,1,2);
            currentInstance1.transform.rotation = Quaternion.Euler(0,0,0);
            currentInstance1.transform.localScale = new Vector3(1,1,1);
            rb1 = currentInstance1.AddComponent<Rigidbody>();
        }
        if (currentInstance1.name=="Prism(Clone)")
        {
            currentInstance1.transform.position = new Vector3(-0.75f,1.5f,2);
            currentInstance1.transform.rotation = Quaternion.Euler(0,0,0);
            currentInstance1.transform.localScale = new Vector3(3,3,3);
            rb1 = currentInstance1.AddComponent<Rigidbody>();
        }
        if (currentInstance1.name=="Torus(Clone)")
        {
            currentInstance1.transform.position = new Vector3(-1,0.75f,2);
            currentInstance1.transform.rotation = Quaternion.Euler(0,0,0);
            currentInstance1.transform.localScale = new Vector3(1.75f,2.5f,1.75f);
            rb1 = currentInstance1.AddComponent<Rigidbody>();
        }



        currentInstance2=GameObject.Instantiate(prefabs_mat2.Sample());
        //hard code some objects that are required
        if (currentInstance2.name=="Capsule(Clone)")
        {
            currentInstance2.transform.position = new Vector3(1.5f,1.5f,2);
            currentInstance2.transform.rotation = Quaternion.Euler(0,0,30);
            currentInstance2.transform.localScale = new Vector3(1.5f,1.5f,1.5f);
            rb2 = currentInstance2.AddComponent<Rigidbody>();
        }
        if (currentInstance2.name=="Cube(Clone)")
        {
            currentInstance2.transform.position = new Vector3(1.25f,1.5f,2);
            currentInstance2.transform.rotation = Quaternion.Euler(0,0,30);
            currentInstance2.transform.localScale = new Vector3(1,3,2);
            rb2 = currentInstance2.AddComponent<Rigidbody>();
        }
        if (currentInstance2.name=="Pipe(Clone)")
        {
            currentInstance2.transform.position = new Vector3(1.75f,1.9f,2);
            currentInstance2.transform.rotation = Quaternion.Euler(0,0,40);
            currentInstance2.transform.localScale = new Vector3(1,1.5f,1);
            rb2 = currentInstance2.AddComponent<Rigidbody>();
        }
        if (currentInstance2.name=="Torus(Clone)")
        {
            currentInstance2.transform.position = new Vector3(1.5f,2,2);
            currentInstance2.transform.rotation = Quaternion.Euler(0,0,-70);
            currentInstance2.transform.localScale = new Vector3(2,2,2);
            rb2 = currentInstance2.AddComponent<Rigidbody>();
        }

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