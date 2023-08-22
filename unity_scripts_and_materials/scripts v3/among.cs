using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;


[Serializable]
[AddRandomizerMenu("Perception/among")]
public class among : Randomizer
{
    // fixed variables
    public GameObjectParameter prefabs1;
    private GameObject currentInstance1;
    public GameObjectParameter prefabs2;
    private GameObject currentInstance2;
    public MaterialParameter materials1;
    private Material current_material1;
    public MaterialParameter materials2;
    private Material current_material2;
    public Camera mainCamera;

    //public variables to change
    public FloatParameter objectScale1;
    public int minNumObjects1;
    public int maxNumObjects1;
    public FloatParameter objectScale2;
    public int minNumObjects2;
    public int maxNumObjects2;
    private int objectCount1;
    private int objectCount2;
    public FloatParameter spawnRadius1;
    public FloatParameter spawnCollisionCheckRadius1;   
    public FloatParameter spawnRadius2;
    public FloatParameter spawnCollisionCheckRadius2;   
    public Vector3Parameter objectRotation1;
    public Vector3Parameter objectRotation2;
    public FloatParameter cameraDepthZ;
    public FloatParameter cameraCircleRadius;
    public Vector3Parameter cameraRotation;
    public FloatParameter cameraFOV ;    
    
    protected override void OnIterationStart()
    {
        objectCount1 = Random.Range(minNumObjects1,maxNumObjects1);   
        currentInstance1 = GameObject.Instantiate(prefabs1.Sample());
        currentInstance1.transform.position = new Vector3(0,0,0);
        for (int loop=0;loop<objectCount1;loop++)
        {
            Vector3 spawnPoint = currentInstance1.transform.position + Random.insideUnitSphere * spawnRadius1.Sample();
            if (!Physics.CheckSphere(spawnPoint,spawnCollisionCheckRadius1.Sample()))
            {
                GameObject spawnedObject = GameObject.Instantiate(currentInstance1,spawnPoint,Random.rotation);
                spawnedObject.transform.localScale = Vector3.one * objectScale1.Sample();
                spawnedObject.transform.rotation = Quaternion.Euler(objectRotation1.Sample());
            }
        }

        objectCount2 = Random.Range(minNumObjects2,maxNumObjects2);   
        currentInstance2 = GameObject.Instantiate(prefabs2.Sample());
        currentInstance2.transform.position = new Vector3(0,0,0);
        for (int loop=0;loop<objectCount2;loop++)
        {
            Vector3 spawnPoint = currentInstance2.transform.position + Random.insideUnitSphere * spawnRadius2.Sample();
            if (!Physics.CheckSphere(spawnPoint,spawnCollisionCheckRadius2.Sample()))
            {
                GameObject spawnedObject = GameObject.Instantiate(currentInstance2,spawnPoint,Random.rotation);
                spawnedObject.transform.localScale = Vector3.one * objectScale2.Sample();
                spawnedObject.transform.rotation = Quaternion.Euler(objectRotation2.Sample());
            }
        }

        Vector2 cameraCirclePos = Random.insideUnitCircle * cameraCircleRadius.Sample();
        mainCamera.transform.position = new Vector3(cameraCirclePos.x,cameraCirclePos.y,cameraDepthZ.Sample());
        mainCamera.transform.rotation = Quaternion.Euler(cameraRotation.Sample());
        mainCamera.fieldOfView = cameraFOV.Sample();

        // dont need to change -  for non container objects only 
        GameObject[] GameObjects = (GameObject.FindObjectsOfType<GameObject>() as GameObject[]);
        for (int i = 0; i < GameObjects.Length; i++)
        {
            if (GameObjects[i].name==currentInstance1.name || GameObjects[i].name==currentInstance1.name+"(Clone)") 
            {
                if (!GameObjects[i].name.StartsWith("person") && !GameObjects[i].name.StartsWith("track") && !GameObjects[i].name.StartsWith("tunnel"))
                {
                    MeshRenderer[] meshRenderers1 = GameObjects[i].GetComponentsInChildren<MeshRenderer>();
                    foreach (MeshRenderer meshRenderer in meshRenderers1) 
                    {
                        current_material1 = materials1.Sample(); // assign a random material to each of the mesh renderers from the list of materials selected in the UI
                        meshRenderer.material = current_material1;        
                        MeshCollider meshCollider = meshRenderer.gameObject.AddComponent<MeshCollider>(); // only for NON CONTAINER objects !!! assign a meshcollider to eahc of the mesh renderers
                        meshCollider.convex = true;
                    }        
                    for (int j = 0; j < materials1.GetCategoryCount(); j++) // assign a random color to each of the materials in the materials list, doing it like this in a separate loop is more efficient since you set a color only once for each material
                    {
                        materials1.GetCategory(j).color = Random.ColorHSV();
                    }  
                }
            }

            else if (GameObjects[i].name==currentInstance2.name || GameObjects[i].name==currentInstance2.name+"(Clone)") 
            {
                if (!GameObjects[i].name.StartsWith("person") && !GameObjects[i].name.StartsWith("track") && !GameObjects[i].name.StartsWith("tunnel"))
                {
                    MeshRenderer[] meshRenderers2 = GameObjects[i].GetComponentsInChildren<MeshRenderer>();
                    foreach (MeshRenderer meshRenderer in meshRenderers2) 
                    {
                        current_material2 = materials2.Sample(); // assign a random material to each of the mesh renderers from the list of materials selected in the UI
                        meshRenderer.material = current_material2;        
                        MeshCollider meshCollider = meshRenderer.gameObject.AddComponent<MeshCollider>(); // only for NON CONTAINER objects !!! assign a meshcollider to eahc of the mesh renderers
                        meshCollider.convex = true;
                    }        
                    for (int j = 0; j < materials2.GetCategoryCount(); j++) // assign a random color to each of the materials in the materials list, doing it like this in a separate loop is more efficient since you set a color only once for each material
                    {
                        materials2.GetCategory(j).color = Random.ColorHSV();
                    }  
                }
            }
        }
    }
    protected override void OnIterationEnd()
    {
        GameObject[] GameObjects = (GameObject.FindObjectsOfType<GameObject>() as GameObject[]);
        for (int i = 0; i < GameObjects.Length; i++)
        {
            if (GameObjects[i].name==currentInstance1.name || GameObjects[i].name==currentInstance1.name+"(Clone)" || GameObjects[i].name==currentInstance2.name || GameObjects[i].name==currentInstance2.name+"(Clone)")
            {
                GameObject.Destroy(GameObjects[i]);
            }
        }
    }
}