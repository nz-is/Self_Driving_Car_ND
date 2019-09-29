# self-driving-car

In this repository I will share the **source code** of all the projects of **[Udacity Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013)**.


## Overview

### Projects

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./CarND_P1/examples/overview.gif" alt="Overview" width="60%" height="60%">
           <br>P1: Basic Lane Finding
           <br><a href="./CarND_P1" name="p1_code">(code)</a>
      </p>
    </th>
        <th><p align="center">
           <br>P2: Traffic Signs
           <br><a href="./project_2_traffic_sign_classifier" name="p2_code">TODO</a>
        </p>
    </th>
       <th><p align="center">
           <br>P3: Behavioral Cloning
           <br><a href="./project_3_behavioral_cloning" name="p3_code">TODO</a>
        </p>
    </th>
        <th><p align="center">
           <br>P4: Adv. Lane Finding
           <br><a href="./project_4_advanced_lane_finding" name="p4_code">TODO</a>
        </p>
    </th>
  </tr>
  <tr>
    <th><p align="center">
           <br>P5: Vehicle Detection
           <br><a href="./project_5_vehicle_detection" name="p5_code">TODO</a>
        </p>
    </th>
        <th><p align="center">
           <br>P6: Ext. Kalman Filter
           <br><a href="./project_6_extended_kalman_filter" name="p6_code">TODO</a>
        </p>
    </th>
    <th><p align="center">
           <br>P7: Unsc. Kalman Filter
           <br><a href="./project_7_unscented_kalman_filter" name="p7_code">TODO</a>
        </p>
    </th>
    <th><p align="center">
           <br>P8: Kidnapped Vehicle
           <br><a href="./project_8_kidnapped_vehicle" name="p8_code">(code)</a>
        </p>
    </th>
  </tr>
  <tr>
    <th><p align="center">
           <br>P9: PID Controller
           <br><a href="" name="p9_code">TODO</a>
        </p>
    </th>
    <th><p align="center">
           <br>P10: MPC Controller
           <br><a href="./project_10_MPC_control" name="p10_code">TODO</a>
        </p>
    </th>
   <th><p align="center">
           <br>P11: Path Planning
           <br><a href="./project_11_path_planning" name="p11_code">TODO</a>
        </p>
    </th>
    <th><p align="center">
           <br>P12: Road Segmentation
           <br><a href="./project_12_road_segmentation" name="p12_code">TODO</a>
        </p>
    </th>
  </tr>
</table>
---
## Table of Contents

#### [P1 - Detecting Lane Lines (basic)](project_1_lane_finding_basic)
 - **Summary:** Detected highway lane lines on a video stream. Used OpencV image analysis techniques to identify lines, including Hough Transforms and Canny edge detection.
 - **Keywords:** Computer Vision

#### [P2 - Traffic Sign Classification](project_2_traffic_sign_classifier)
 - **Summary:** Built and trained a deep neural network to classify traffic signs, using TensorFlow. Experimented with different network architectures. Performed image pre-processing and validation to guard against overfitting.
 - **Keywords:** Deep Learning, TensorFlow, Computer Vision

#### [P3 - Behavioral Cloning](project_3_behavioral_cloning)
 - **Summary:** Built and trained a convolutional neural network for end-to-end driving in a simulator, using TensorFlow and Keras. Used optimization techniques such as regularization and dropout to generalize the network for driving on multiple tracks.
 - **Keywords:** Deep Learning, Keras, Convolutional Neural Networks

#### [P4 - Advanced Lane Finding](project_4_advanced_lane_finding)
 - **Summary:** Built an advanced lane-finding algorithm using distortion correction, image rectification, color transforms, and gradient thresholding. Identified lane curvature and vehicle displacement. Overcame environmental challenges such as shadows and pavement changes.
 - **Keywords:** Computer Vision, OpenCV

#### [P5 - Vehicle Detection and Tracking](project_5_vehicle_detection)
 - **Summary:** Created a vehicle detection and tracking pipeline with OpenCV, histogram of oriented gradients (HOG), and support vector machines (SVM). Implemented the same pipeline using a deep network to perform detection. Optimized and evaluated the model on video data from a automotive camera taken during highway driving.
 - **Keywords:** Computer Vision, Deep Learning, OpenCV

 #### [P6 - Extended Kalman Filter](project_6_extended_kalman_filter)
 - **Summary:** Implement the extended Kalman filter in C++. Simulated lidar and radar measurements are used to detect a bicycle that travels around your vehicle. Kalman filter, lidar measurements and radar measurements are used to track the bicycle's position and velocity.
 - **Keywords:** C++, Kalman Filter

 #### [P7 - Unscented Kalman Filter](project_7_unscented_kalman_filter)
 - **Summary:**  Utilize an Unscented Kalman Filter to estimate the state of a moving object of interest with noisy lidar and radar measurements. Kalman filter, lidar measurements and radar measurements are used to track the bicycle's position and velocity.
 - **Keywords:** C++, Kalman Filter

  #### [P8 - Kidnapped Vehicle](project_8_kidnapped_vehicle)
 - **Summary:** Your robot has been kidnapped and transported to a new location! Luckily it has a map of this location, a (noisy) GPS estimate of its initial location, and lots of (noisy) sensor and control data. In this project you will implement a 2 dimensional particle filter in C++. Your particle filter will be given a map and some initial localization information (analogous to what a GPS would provide). At each time step your filter will also get observation and control data.
 - **Keywords:** C++, Particle Filter

 #### [P9 - PID Control](project_9_PID_control)
 - **Summary:** Implement a PID controller for keeping the car on track by appropriately adjusting the steering angle.
 - **Keywords:** C++, PID Controller

#### [P10 - MPC Control](project_10_MPC_control)
- **Summary:** Implement an MPC controller for keeping the car on track by appropriately adjusting the steering angle. Differently from previously implemented PID controller, MPC controller has the ability to anticipate future events and can take control actions accordingly. Indeed, future time steps are taking into account while optimizing current time slot.
- **Keywords:** C++, MPC Controller

#### [P11 - Path Planning](project_11_path_planning)
- **Summary:** The goal in this project is to build a path planner that is able to create smooth, safe trajectories for the car to follow. The highway track has other vehicles, all going different speeds, but approximately obeying the 50 MPH speed limit. The car transmits its location, along with its sensor fusion data, which estimates the location of all the vehicles on the same side of the road.
- **Keywords:** C++, Path Planning

#### [P12 - Road Segmentation](project_12_road_segmentation)
- **Summary:** Implement the road segmentation using a fully-convolutional network.
- **Keywords:** Python, TensorFlow, Semantic Segmentation
