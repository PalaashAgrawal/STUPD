using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;


[Serializable]
[AddRandomizerMenu("Perception/into")]
public class into : Randomizer
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
    public Vector3Parameter objectPos1;
    public Vector3Parameter objectPos2;
    public FloatParameter objectScale1;
    public FloatParameter objectScale2;
    public Vector3Parameter objectRotation1;
    public Vector3Parameter objectRotation2;
    public FloatParameter cameraSphereRadius;
    public FloatParameter cameraSphereLimit;
    public Vector3Parameter cameraRandomLookAtOffset;
    public FloatParameter cameraRotationZ;
    public FloatParameter cameraFOV;
    private float object1Y;
    public FloatParameter speed; 
    private float timeCounter; 
    // private Rigidbody object1rb;

    protected override void OnIterationStart()
    {
        timeCounter = 0f; 
        Vector3 pos1 = objectPos1.Sample(); // object pos 1
        Vector3 pos2 = objectPos2.Sample(); // object pos 2
        object1Y = pos1.y;
        currentInstance1=GameObject.Instantiate(prefabs1.Sample(), pos1, Quaternion.identity);
        currentInstance2=GameObject.Instantiate(prefabs2.Sample(), pos2, Quaternion.identity);
        currentInstance1.transform.localScale = Vector3.one * objectScale1.Sample();
        currentInstance1.transform.rotation = Quaternion.Euler(objectRotation1.Sample());
        currentInstance2.transform.localScale = Vector3.one * objectScale2.Sample();
        currentInstance2.transform.rotation = Quaternion.Euler(objectRotation2.Sample());

        mainCamera.transform.position = Random.onUnitSphere * cameraSphereRadius.Sample();
        float cameraLimit = cameraSphereLimit.Sample();
        while (mainCamera.transform.position.y>cameraLimit | mainCamera.transform.position.y<-cameraLimit)
        {
            mainCamera.transform.position = Random.onUnitSphere * cameraSphereRadius.Sample();
        }
        Vector3 lookAtPoint = cameraRandomLookAtOffset.Sample();
        mainCamera.transform.LookAt(lookAtPoint);
        float cameraRotZ = cameraRotationZ.Sample();
        Quaternion newRotation = Quaternion.Euler(
            mainCamera.transform.rotation.eulerAngles.x,
            mainCamera.transform.rotation.eulerAngles.y,
            cameraRotZ
        );
        mainCamera.transform.rotation = newRotation;
        mainCamera.fieldOfView = cameraFOV.Sample();

        // dont need to change -  for non container objects only 
        if (!currentInstance1.name.StartsWith("person") && !currentInstance1.name.StartsWith("track"))
        {
            MeshRenderer[] meshRenderers1 = currentInstance1.GetComponentsInChildren<MeshRenderer>();
            foreach (MeshRenderer meshRenderer in meshRenderers1) 
            {
                current_material1 = materials1.Sample(); // assign a random material to each of the mesh renderers from the list of materials selected in the UI
                meshRenderer.material = current_material1;        
                // MeshCollider meshCollider = meshRenderer.gameObject.AddComponent<MeshCollider>(); // only for NON CONTAINER objects !!! assign a meshcollider to eahc of the mesh renderers
                // meshCollider.convex = true;
            }        
            for (int i = 0; i < materials1.GetCategoryCount(); i++) // assign a random color to each of the materials in the materials list, doing it like this in a separate loop is more efficient since you set a color only once for each material
            {
                float r = Random.Range(0f, 1f);
                float g = Random.Range(0f, 1f);
                float b = Random.Range(0f, 1f);
                Color randomColor = new Color(r, g, b);

                materials1.GetCategory(i).color = randomColor;
            }
        }

        if (!currentInstance2.name.StartsWith("person") && !currentInstance2.name.StartsWith("track"))
        {
            MeshRenderer[] meshRenderers2 = currentInstance2.GetComponentsInChildren<MeshRenderer>();
            foreach (MeshRenderer meshRenderer in meshRenderers2) 
            {
                current_material2 = materials2.Sample(); // assign a random material to each of the mesh renderers from the list of materials selected in the UI
                meshRenderer.material = current_material2;        
                // MeshCollider meshCollider = meshRenderer.gameObject.AddComponent<MeshCollider>(); // only for NON CONTAINER objects !!! assign a meshcollider to eahc of the mesh renderers
                // meshCollider.convex = true;
            }        
            for (int i = 0; i < materials2.GetCategoryCount(); i++) // assign a random color to each of the materials in the materials list, doing it like this in a separate loop is more efficient since you set a color only once for each material
            {
                float r = Random.Range(0f, 1f);
                float g = Random.Range(0f, 1f);
                float b = Random.Range(0f, 1f);
                Color randomColor = new Color(r, g, b);

                materials2.GetCategory(i).color = randomColor;
            }
        }
    }

    protected override void OnUpdate()
    {
        timeCounter += Time.deltaTime * speed.Sample();
        float x1 = currentInstance1.transform.position.x;
        float y1 = object1Y-timeCounter;
        float z1 = currentInstance1.transform.position.z;

        if (y1 > currentInstance2.transform.position.y)
        {
            y1 = object1Y - timeCounter;
            currentInstance1.transform.position = new Vector3(x1, y1, z1);
        }
        else
        {
            currentInstance1.transform.position = new Vector3(x1, currentInstance2.transform.position.y , z1);
        }
    }


    protected override void OnIterationEnd()
    {
        GameObject.Destroy(currentInstance1);
        GameObject.Destroy(currentInstance2);
    }
}
