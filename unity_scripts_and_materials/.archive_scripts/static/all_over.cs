using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;

[Serializable]
[AddRandomizerMenu("Perception/all_over")]
public class all_over : Randomizer
{
    // prefabs selection
    public GameObjectParameter prefabs_mat1;
    private GameObject currentInstance;
    private GameObject instanceType;
    private int objectCount;
    public float spawnRadius=3;
    public float spawnCollisionCheckRadius=0.5f;   
    // camera selection 
    public Camera mainCamera;
    public Vector3Parameter cameraPosition;
    // materials selection
    public Material material1;
    public Material material2;

    protected override void OnIterationStart()
    {   
        objectCount = Random.Range(3,10);
        currentInstance = GameObject.Instantiate(prefabs_mat1.Sample());
        currentInstance.transform.position = new Vector3(0,0,0);
        for (int loop=0;loop<objectCount;loop++)
        {
            Vector3 spawnPoint = currentInstance.transform.position + Random.insideUnitSphere * spawnRadius;
            if (!Physics.CheckSphere(spawnPoint,spawnCollisionCheckRadius))
            {
                GameObject.Instantiate(currentInstance,spawnPoint,Random.rotation);
            }
        }

        // camera randomisation code
        mainCamera.transform.position = cameraPosition.Sample();
        mainCamera.transform.LookAt(new Vector3(0,0,0)); // can set a manual point to look at

        // material randomisation code
        material1.color = Random.ColorHSV();
    }

    protected override void OnIterationEnd()
    {
        GameObject[] GameObjects = (GameObject.FindObjectsOfType<GameObject>() as GameObject[]);
        for (int i = 0; i < GameObjects.Length; i++)
        {
            if (GameObjects[i].name==currentInstance.name || GameObjects[i].name==currentInstance.name+"(Clone)")
            {
                GameObject.Destroy(GameObjects[i]);
            }
        }
    }
}