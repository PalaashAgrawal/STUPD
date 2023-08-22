using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/beside")]
public class beside : Randomizer
{
    // prefabs selection
    public GameObjectParameter prefabs_mat1;
    public GameObjectParameter prefabs_mat2;
    private GameObject currentInstance1;
    private GameObject currentInstance2;
    private float random_x;
    private float random_z;
    private float position_switch_operator;
    // camera selection 
    public Camera mainCamera;
    public Vector3Parameter cameraPosition;
    // materials selection
    public Material material1;
    public Material material2;

    protected override void OnIterationStart()
    {   
        // object variables code
        random_x = Random.Range(1.5f,5);
        random_z = Random.Range(-2.5f,2.5f);
        position_switch_operator = Random.Range(-2,2);
        while (position_switch_operator==0)
        {
            position_switch_operator = Random.Range(-2,2);
        }

        Debug.Log(position_switch_operator);

        if (position_switch_operator>0)
        {
            currentInstance1=GameObject.Instantiate(prefabs_mat1.Sample());
            currentInstance1.transform.position = new Vector3(random_x,0,random_z);
            currentInstance1.transform.rotation = Random.rotation;
            
            currentInstance2=GameObject.Instantiate(prefabs_mat2.Sample());
            currentInstance2.transform.position = new Vector3(-random_x,0,random_z);
            currentInstance2.transform.rotation = Random.rotation;
        }

        else
        {
            currentInstance1=GameObject.Instantiate(prefabs_mat1.Sample());
            currentInstance1.transform.position = new Vector3(-random_x,0,random_z);
            currentInstance1.transform.rotation = Random.rotation;
            
            currentInstance2=GameObject.Instantiate(prefabs_mat2.Sample());
            currentInstance2.transform.position = new Vector3(random_x,0,random_z);
            currentInstance2.transform.rotation = Random.rotation;
        }

        // camera randomisation code
        mainCamera.transform.position = cameraPosition.Sample();
        // mainCamera.transform.LookAt(currentInstance1.transform); //look at gameobject position
        mainCamera.transform.LookAt(new Vector3(0,0,0)); // can set a manual point to look at

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