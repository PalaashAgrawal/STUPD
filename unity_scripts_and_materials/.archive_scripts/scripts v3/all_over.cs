using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;


[Serializable]
[AddRandomizerMenu("Perception/all over")]
public class all_over : Randomizer
{
    // fixed variables
    public GameObjectParameter prefabs;
    private GameObject currentInstance;
    public MaterialParameter materials1;
    private Material current_material1;
    public Camera mainCamera;

    //public variables to change
    public FloatParameter objectScale;
    public int minNumObjects;
    public int maxNumObjects;
    private int objectCount;
    public FloatParameter spawnRadius;
    public FloatParameter spawnCollisionCheckRadius;   
    public Vector3Parameter objectRotation;
    public FloatParameter cameraDepthZ;
    public FloatParameter cameraCircleRadius;
    public Vector3Parameter cameraRotation;
    public FloatParameter cameraFOV ;    
    
    protected override void OnIterationStart()
    {
        objectCount = Random.Range(minNumObjects,maxNumObjects);   
        currentInstance = GameObject.Instantiate(prefabs.Sample());
        currentInstance.transform.position = new Vector3(0,0,0);
        for (int loop=0;loop<objectCount;loop++)
        {
            Vector3 spawnPoint = currentInstance.transform.position + Random.insideUnitSphere * spawnRadius.Sample();
            if (!Physics.CheckSphere(spawnPoint,spawnCollisionCheckRadius.Sample()))
            {
                GameObject spawnedObject = GameObject.Instantiate(currentInstance,spawnPoint,Random.rotation);
                spawnedObject.transform.localScale = Vector3.one * objectScale.Sample();
                spawnedObject.transform.rotation = Quaternion.Euler(objectRotation.Sample());
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