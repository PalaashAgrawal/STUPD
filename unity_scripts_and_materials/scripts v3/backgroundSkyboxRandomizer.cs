using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using System;
using Random = UnityEngine.Random;

[Serializable]
[AddRandomizerMenu("Perception/background Skybox Randomizer")]
public class backgroundSkyboxRandomizer : Randomizer
{
    public Material[] skyboxMaterials;
    public FloatParameter rotation;

    protected override void OnIterationStart()
    {
        RenderSettings.skybox.SetFloat("_Rotation",rotation.Sample());
    }

    protected override void OnIterationEnd()
    {
        Material selectedMaterial = skyboxMaterials[Random.Range(0, skyboxMaterials.Length)];
        // Set the selected skybox material as the active skybox material
        RenderSettings.skybox = selectedMaterial;
    }
}


// using UnityEngine;
// using UnityEngine.UI;
// using UnityEngine.Perception.Randomization.Parameters;
// using UnityEngine.Perception.Randomization.Randomizers;
// using System;
// using System.IO;
// using System.Collections;
// using System.Collections.Generic;
// using Random=UnityEngine.Random;

// [Serializable]
// [AddRandomizerMenu("Perception/background Skybox Randomizer")]
// public class backgroundSkyboxRandomizer : Randomizer
// {
//     public Material[] skyboxMaterials;
//     public FloatParameter rotationX;
//     public FloatParameter rotationY;
//     public FloatParameter rotationZ;

//     protected override void OnIterationStart()
//     {
//         Material selectedMaterial = skyboxMaterials[Random.Range(0, skyboxMaterials.Length)];
//         // Set the selected skybox material as the active skybox material
//         RenderSettings.skybox = selectedMaterial;
//     }
//     protected virtual void OnStartRunning()
//     {
//         float rX = rotationX.Sample();
//         float rY = rotationY.Sample();
//         float rZ = rotationZ.Sample();
//         RenderSettings.skybox.SetFloat("_RotationX", rX);
//         RenderSettings.skybox.SetFloat("_RotationY", rY);
//         RenderSettings.skybox.SetFloat("_RotationZ", rZ);
//     }
// }