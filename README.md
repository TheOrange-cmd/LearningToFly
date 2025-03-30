# MAIN README

Welcome to the repository of 'Embedded Monocular Obstacle Avoidance in MAVs Using Extreme Knowledge Distillation' by D.J. Rugge, T.M. van Dam, M. Frans, W. van Mildert, T.B.M. van Santen, J. Slagmolen. This repository is a fork of the specially prepared fork of the paparazzi UAV project for the MSc course 'AE4317 - Autonomous Flight of Micro Air Vehicles' at the TU Delft. For an overview of paparazzi, the MAVlab fork, and the course, the reader is referred to the course manual, found on https://tudelft.github.io/coursePaparazzi/. Our module showed great flying speeds, winning the distance competition by 100 meters compared to the runner-up, but also caused significant collisions. Further testing is required to find the exact point of failure - whether it stems from the sensing, the steering, or both! We ended up reaching second place in the course competition, and learned a lot from the project. If you're reading this, make sure to first read our report, which is also included in this repo under report.pdf. It contains all the high level, and some low level, info, as well as some background, related work, and ideas for future work. 

The repository looks a little messy due to all the paparazzi related files necessary for actually flying the drone. But to point you in the right direction, our contributions to this fork can be found in the following folders/files:

```
FrontCamDetector
BottomCamDetector
SmallConvNetwork
sw/airborne/modules/computervision/obstacle_detection 
sw/airborne/modules/computervision/obstacle_avoidance
utils
conf/airframes/tudelft/bebop_obstacle_avoid
```

Respectively, they contain:

- Files related to the MiDaS based hyper distilled obstacle detection model described in the report.
- Files related to a model that performs boundary detection in the Cyberzoo, more on this below.
- files related to a model based on the bounding box approach discussed in the report. 
- C and header files related to the custom paparazzi module for obstacle detection and boundary detection.
- C and header files related to the custom paparazzi module for obstacle avoidance using the detections found by the detection module.
- Two bash scripts to convert and benchmark ONNX models for the drone.
- An airframe file heavily based on the airframe file provided by the course - not very interesting! 

However, the report does not contain a discussion on the code in the BottomCamDetector folder, as we were unable to integrate it well with the avoidance code. It is a simple model which is trained to determine whether the drone is inside or outside the Cyberzoo boundary, using manually labeled images from the bottom camera showing either the fake grass area (sometimes partially or fully covered by mats or sheets), or showing the boundary of the cyberzoo in parts of the image. The model is a simple binary classifier which can thus determine if the drone is inside or outside the boundary, but steering with this limited information proved difficult! As the trained classifier only has an inference time of about 3 ms including preprocessing, it could be a little more complex with a more informative output like which region is safe to fly towards (or not). 
