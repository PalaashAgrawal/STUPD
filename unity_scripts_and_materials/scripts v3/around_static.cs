using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;


[Serializable]
[AddRandomizerMenu("Perception/around static")]
public class around_static : Randomizer
{
    // fixed variables
    public GameObjectParameter prefabs1;
    public GameObjectParameter prefabs2;
    private GameObject currentInstance1;
    private GameObject currentInstance2;
    public MaterialParameter materials1;
    private Material current_material1;
    public MaterialParameter materials2;
    private Material current_material2;
    public Camera mainCamera;

    //public variables to change
    public Vector3Parameter centreObjectPos;
    // public float minObjectPosRadius1;
    // public float minObjectPosRadius2;
    public FloatParameter spawnRadius;
    public FloatParameter spawnCollisionCheckRadius;  
    public FloatParameter objectScale1;
    public FloatParameter objectScale2;
    public int minNumObjects;
    public int maxNumObjects;
    private int objectCount;
    public Vector3Parameter objectRotation1;
    public Vector3Parameter objectRotation2;   
    public FloatParameter cameraDepthZ;
    public FloatParameter cameraCircleRadius;
    public Vector3Parameter cameraRotation;
    public FloatParameter cameraFOV ;

    protected override void OnIterationStart()
    {
        Vector3 pos1 = centreObjectPos.Sample(); // centre object pos
        currentInstance1 = GameObject.Instantiate(prefabs1.Sample(), pos1, Quaternion.identity);
        currentInstance1.transform.localScale = Vector3.one * objectScale1.Sample();
        currentInstance1.transform.rotation = Quaternion.Euler(objectRotation1.Sample());

        currentInstance2 = GameObject.Instantiate(prefabs2.Sample(), new Vector3(1000,0,0),Quaternion.identity);
        objectCount = Random.Range(minNumObjects,maxNumObjects); 

        float angleStep = 2 * Mathf.PI / objectCount;
        float currentAngle = Random.value * 2 * Mathf.PI;

        for (int loop = 0; loop < objectCount; loop++)
        {
            Vector3 circle3D = new Vector3(Mathf.Cos(currentAngle), Mathf.Sin(currentAngle), 0) * spawnRadius.Sample();
            Vector3 spawnPoint = currentInstance1.transform.position + circle3D;
            if (!Physics.CheckSphere(spawnPoint, spawnCollisionCheckRadius.Sample()))
            {
                GameObject spawnedObject = GameObject.Instantiate(currentInstance2, spawnPoint, Quaternion.identity);
                spawnedObject.transform.localScale = Vector3.one * objectScale2.Sample();
                spawnedObject.transform.rotation = Quaternion.Euler(objectRotation2.Sample());
            }

            currentAngle += angleStep;
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