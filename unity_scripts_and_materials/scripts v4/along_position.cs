using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;


[Serializable]
[AddRandomizerMenu("Perception/along position")]
public class along_position : Randomizer
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
    public Vector3Parameter object1Pos;
    public Vector3Parameter object2Pos;
    public FloatParameter objectScale1;
    public Vector3Parameter objectScale2;
    public Vector3Parameter objectRotation1;
    public Vector3Parameter objectRotation2;   
    public int minNumObjects;
    public int maxNumObjects;
    private int objectCount;
    private float randomX;
    public FloatParameter cameraSphereRadius;
    public FloatParameter cameraSphereLimit;
    public Vector3Parameter cameraRandomLookAtOffset;
    public FloatParameter cameraFOV;

    protected override void OnIterationStart()
    {
        // static Vector3 RandomPointOnLine(float x, float y, float z)
        // {
        //     Vector3 point = new Vector3(x,y,z);
        //     return point;
        // }

        randomX = 0;
        objectCount = Random.Range(minNumObjects,maxNumObjects);   
        Vector3 pos1 = object1Pos.Sample(); // object pos 1
        Vector3 pos2 = object2Pos.Sample(); // object pos 1
        currentInstance1=GameObject.Instantiate(prefabs1.Sample(), pos1, Quaternion.identity);
        currentInstance2=GameObject.Instantiate(prefabs2.Sample(), pos2, Quaternion.identity);

        currentInstance1.transform.localScale = Vector3.one * objectScale1.Sample();
        currentInstance2.transform.localScale = objectScale2.Sample();
        currentInstance1.transform.rotation = Quaternion.Euler(objectRotation1.Sample());
        currentInstance2.transform.rotation = Quaternion.Euler(objectRotation2.Sample());

        for (int loop=0;loop<objectCount;loop++)
        {
            float random_step = Random.Range(1.5f,2);
            randomX += random_step;
            Vector3 currentPos1 = currentInstance1.transform.position;
            Vector3 spawnPoint = new Vector3(currentPos1.x+randomX,currentPos1.y,currentPos1.z);
            GameObject.Instantiate(currentInstance1,spawnPoint,Random.rotation);
        }

        mainCamera.transform.position = Random.onUnitSphere * cameraSphereRadius.Sample();
        while (mainCamera.transform.position.y<0)
        {
            mainCamera.transform.position = Random.onUnitSphere * cameraSphereRadius.Sample();
        }
        Vector3 lookAtPoint = cameraRandomLookAtOffset.Sample();
        mainCamera.transform.LookAt(lookAtPoint);
        mainCamera.fieldOfView = cameraFOV.Sample();

        // dont need to change -  for non container objects only 
        GameObject[] GameObjects = (GameObject.FindObjectsOfType<GameObject>() as GameObject[]);
        for (int i = 0; i < GameObjects.Length; i++)
        {
            if (GameObjects[i].name==currentInstance1.name || GameObjects[i].name==currentInstance1.name+"(Clone)") 
            {
                if (!GameObjects[i].name.StartsWith("person") && !GameObjects[i].name.StartsWith("track"))
                {
                    MeshRenderer[] meshRenderers1 = GameObjects[i].GetComponentsInChildren<MeshRenderer>();
                    foreach (MeshRenderer meshRenderer in meshRenderers1) 
                    {
                        current_material1 = materials1.Sample(); // assign a random material to each of the mesh renderers from the list of materials selected in the UI
                        meshRenderer.material = current_material1;        
                        // MeshCollider meshCollider = meshRenderer.gameObject.AddComponent<MeshCollider>(); // only for NON CONTAINER objects !!! assign a meshcollider to eahc of the mesh renderers
                        // meshCollider.convex = true;
                    }        
                    for (int j = 0; j < materials1.GetCategoryCount(); j++) // assign a random color to each of the materials in the materials list, doing it like this in a separate loop is more efficient since you set a color only once for each material
                    {
                        float r = Random.Range(0f, 1f);
                        float g = Random.Range(0f, 1f);
                        float b = Random.Range(0f, 1f);
                        Color randomColor = new Color(r, g, b);

                        materials1.GetCategory(j).color = randomColor;
                    }
                }
            }

            else if (GameObjects[i].name==currentInstance2.name || GameObjects[i].name==currentInstance2.name+"(Clone)") 
            {
                if (!GameObjects[i].name.StartsWith("person") && !GameObjects[i].name.StartsWith("track"))
                {
                    MeshRenderer[] meshRenderers2 = GameObjects[i].GetComponentsInChildren<MeshRenderer>();
                    foreach (MeshRenderer meshRenderer in meshRenderers2) 
                    {
                        current_material2 = materials2.Sample(); // assign a random material to each of the mesh renderers from the list of materials selected in the UI
                        meshRenderer.material = current_material2;        
                        // MeshCollider meshCollider = meshRenderer.gameObject.AddComponent<MeshCollider>(); // only for NON CONTAINER objects !!! assign a meshcollider to eahc of the mesh renderers
                        // meshCollider.convex = true;
                    }        
                    for (int j = 0; j < materials2.GetCategoryCount(); j++) // assign a random color to each of the materials in the materials list, doing it like this in a separate loop is more efficient since you set a color only once for each material
                    {
                        float r = Random.Range(0f, 1f);
                        float g = Random.Range(0f, 1f);
                        float b = Random.Range(0f, 1f);
                        Color randomColor = new Color(r, g, b);

                        materials2.GetCategory(j).color = randomColor;
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