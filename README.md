# Unreal Rooftop Landing


## Install

### Python Instructions

1. Install [conda](https://conda.io/projects/conda/en/latest/) or create a python virtual environment ([Why?](https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c)). I recommend conda for Windows users.
2. Example - `conda create --name unreal python=3.7 && conda activate unreal` 
2. `pip install -e .`


### Other Projects

There are several projects/plugins used to **make** the actual unreal engine world environments. 

1. Custom Fork of AirSim (Unreal Plugin) with noisy LiDAR model and custom handling of semantic segmentation codes - `https://github.com/JeremyBYU/AirSim`
2. Unreal Plugin to generate *classified* airborne points clouds of a level (map/world) - `https://github.com/JeremyBYU/PointCloudGeneratorUE4`
3. Python scripts to analyze classified airborne point clouds of city rooftops and generate *vector* maps - `https://github.com/JeremyBYU/create-map`
4. Unreal Python scripts to randomly generate and place assets on city rooftops (using vector maps) `https://github.com/JeremyBYU/UnrealLanding_UnrealProject`

Note that items (1,2,4) are installed/used inside of the Unreal Project/Editor itself. The Unreal Project Files can be found [here](https://github.com/JeremyBYU/UnrealLanding_UnrealProject).


## Workflow


### Generate Collection Points

First we need to generate "Collection Points" for where the drone will be. To read up on collection points please see `airsimcollect/README.md`.


#### Computer Vision Training Collection Points

Run `poi generate` for polygons and line data to create the Computer Vision Training Set


1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_training.npy -ns 2`
2. `poi generate -m assets/maps/poi-line.geojson -o assets/collectionpoints/collection_points_cv_training.npy -nf 100 -yd 90 -pr 45 45 -pd 0 -rm 100 -ao`

Note that it doesn't overwrite the file, but will append to it (`-ao` flag).

#### Computer Vision Testing Collection Points

~2000 images for RandomWorldSeed3 and RandomManhattanDistribution each. Pitch range is offset by 5. Basically not only is the data different but it captured at different angles.  Probably overkill but still.

1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_cv_test.npy -rm 500 -pr 35 80 -yd 45`


#### Rooftop Landing Collection Points

This generated collection points for Rooftop LIDAR collection. Much smaller collection. Used for testing actual landing. 

1. `poi generate -m assets/maps/point_cloud_map.geojson -o assets/collectionpoints/collection_points_lidar_landing.npy -ho 1000 -rm 1000 -pr 75 75 -pd 0 -yd 90 -rfn class_label`

### Generate Computer Vision Images

Use the following when collecting only scene and segmentation. Be sure to update your airsim settings in your home directory to use the computer vision mode.

`asc collect -c assets/rooftop/config_computervision_train.json`

Use the following when collecting lidar. Be sure to update your airsim settings in your home directory to use the multirotor

`asc collect -c assets/rooftop/config_lidar.json`

Notes - Sometimes the camera takes time to update position, add more time delay than 0.5 seconds. In other words the lidar and vehicle move to a new position but the camera is still in the old position (AirSim bug).

Use the following when collecting only scene and segmentation for testing. Be sure to update your airsim settings in your home directory to use the computer vision mode.

`asc collect -c assets/rooftop/config_computervision_test.json`

## Visualize

`python lidarsegmentation/airsimvis.py -pm -cm segmentation -c assets/rooftop/config.json`

## LiDAR Sensor Settings

```json
        "0": {
          "SensorType": 6,
          "Enabled": true,
          "NumberOfChannels": 16,
          "RotationsPerSecond": 10,
          "PointsPerSecond": 100000,
          "X": 0.46,
          "Y": 0,
          "Z": 0,
          "Roll": 0,
          "Pitch": 0,
          "Yaw": 0,
          "VerticalFOVUpper": -5,
          "VerticalFOVLower": -45,
          "HorizontalFOVStart": -30,
          "HorizontalFOVEnd": 30,
          "DrawDebugPoints": false,
          "DataFrame": "VehicleInertialFrame"
        }
```

## Gather Data Script
`python assets/rooftop/scripts/gatherstats.py -c assets/rooftop/scripts/config_gather_stats.json`

`python assets/rooftop/scripts/gatherstats.py -c assets/rooftop/scripts/config_gather_stats_tx2.json`

* UID 72 is a great example of a failure
  * Aircraft is too far away, point cloud does not pick up on air vents
 


### Dependencies

* shapely (hard)
* shapely-geojson (easy)
* polylidar (easy)
* descartes (easy)
* polylabelfast (easy)
* quaternion (maybe easy)
* AirSim - We just need 2 types

## TODO

* Collect more data (RGB, SEG) from 2 more worlds, 2000 Images each
  * 1 world randomly generated according to the distribution of Manhattan
  * 1 world manually created according to the distribution of Manhattan
  * Send that data to Brian to evaluate model performance for segmentation
* Collect more data of building rooftops. Use the manually created world
  * Each building will capture - Scene, Segmentation, Lidar, records.json. 3 New folders will be created LidarClassified, Polygons, SegmentationPredicted, LabeledScene
    * 4 pictures of each building, 10 meters and 10 meters up, each 4 sides of the roof
    * records.json will need to be modified to identify building name. Make a simple script that integrates poi-roof-lidar.geoson with records.json

  * For each snapshot perform the following and record timing and metrics
    * Every picture will need to sent to DeepLearningModel to provide a prediction. That prediction (picture) will be saved in SegmentationPredicted
      * Record time to predict, and calculate metrics (IOU?)
    * Lidar Point cloud should be projected into ground truth segmented image. True classified point cloud should be saved in LidarClassified
      * Record time to project point cloud
    * Lidar Point cloud should be projected into predicted segmented image. Predicted classified point cloud should be saved in LidarClassified
      * Record time to project point cloud
    * Use polylidar to extract roof polygon for both point clouds. Save polygon into Polygons folder
      * Record time to generate polygon
    * Use polylabel to find greatest inscribed circle. 
      * Record time to find polylabel, Record position 
    * Project polylabel plane and landing zone into scene picture
      * Save picture into labeled scene

  * Do a few 360 point clouds photos?


### What would state_records look like?

  Columns - command, environment, tag, uid, building, time, metric, misc

#### Examples
  * predict_segmentation, laptop, N/A,  0, Building1, 20ms, IOU, N/A
  * classify_point_cloud, laptop, groundtruth, 0, Building1, 20ms, N/A, N/A
  * classify_point_cloud, laptop, predicted, 0, Building1, 20ms, N/A, N/A
  * classify_point_cloud, jetson, predicted, 0, Building1, 20ms, N/A, N/A
  * polylidar, laptop, predicted, 0, Building1, 20ms, N/A, N/A
  * polylidar, jetson, predicted, 0, Building1, 20ms, N/A, N/A
  * polylabel, laptop, predicted, 0, Building1, 20ms, UK, [point,radius]
  * polylabel, laptop, groundtruth, 0, Building1, 20ms, UK, [point,radius]