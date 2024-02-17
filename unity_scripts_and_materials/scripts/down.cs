using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using Random=UnityEngine.Random;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;


[Serializable]
[AddRandomizerMenu("Perception/down")]
public class down : Randomizer
{
    // fixed variables
    public GameObjectParameter prefabs1;
    private GameObject currentInstance1;
    public MaterialParameter materials1;
    private Material current_material1;
    public Camera mainCamera;

    //public variables to change
    public Vector3Parameter objectPos1;
    public FloatParameter objectScale1;
    public Vector3Parameter objectRotation1;
    public FloatParameter cameraDepthZ;
    public FloatParameter cameraCircleRadius;
    public Vector3Parameter cameraRotation;
    public FloatParameter cameraFOV;
    private float objectY;
    public FloatParameter speed; 
    private float timeCounter = 0f; 

    protected override void OnIterationStart()
    {
        timeCounter = 0f; 
        Vector3 pos1 = objectPos1.Sample(); // object pos 1
        currentInstance1=GameObject.Instantiate(prefabs1.Sample(), pos1, Quaternion.identity);
        currentInstance1.transform.localScale = Vector3.one * objectScale1.Sample();
        currentInstance1.transform.rotation = Quaternion.Euler(objectRotation1.Sample());

        Vector2 cameraCirclePos = Random.insideUnitCircle * cameraCircleRadius.Sample();
        mainCamera.transform.position = new Vector3(cameraCirclePos.x,cameraCirclePos.y,cameraDepthZ.Sample());
        mainCamera.transform.rotation = Quaternion.Euler(cameraRotation.Sample());
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
    }

    protected override void OnUpdate() 
    { 
        timeCounter+=Time.deltaTime*speed.Sample(); 
        float x1 = currentInstance1.transform.position.x; 
        float y1 = objectY - timeCounter; 
        float z1 = currentInstance1.transform.position.z; 
        currentInstance1.transform.position = new Vector3(x1,y1,z1); 
    } 

    protected override void OnIterationEnd()
    {
        GameObject.Destroy(currentInstance1);
   }
}