<h1>DriveWise</h1>
<p>DriveWise is an innovative application designed to enhance safety and accountability within taxi rental services by evaluating driver performance in real time. Employing cutting-edge technology, DriveWise assesses drivers based on attentiveness, alertness, and adherence to traffic regulations during journeys. This README provides an overview of DriveWise's functionality, features, and implementation.</p>
<h2>How Does It Work?</h2>
<p>DriveWise uses advanced technology to analyze the driver's behavior and detect potential safety risks. Here's a breakdown of how it works:</p>
<h3>1. Face Detection:</h3>
<p>DriveWise first locates the driver's face in the camera view using specialized algorithms</p>
<h3>2. Keypoint Prediction:</h3>
<p>Once the face is detected, DriveWise predicts key facial landmarks such as eyes, nose, and mouth to track movement and orientation.</p>
<h3>3. Computed Scores:</h3>
<b>- EAR (Eye Aspect Ratio):</b><p> Determines if the driver's eyes are open wide or starting to close, alerting to potential drowsiness.</p>
   <b>- Gaze Score:</b> <p>Monitors the direction of the driver's gaze to ensure focus on the road ahead.</p>
   <b>- Head Pose:</b> <p> Checks for proper head alignment to prevent distracted driving.</p>
   <b>- PERCLOS (PERcentage of CLOSure eye time):</b> <p> Tracks the frequency and duration of eye closures, indicating fatigue or drowsiness.</p>
<h3>4. Driver States:</h3> 
   <b>- Normal:</b> <p>No issues detected, and no alerts are triggered.</p> 
   <b>- Tired:</b> <p>Alerts the driver if signs of fatigue or drowsiness are detected based on eye closure patterns.</p> 
   <b>- Asleep:</b> <p>Warns the driver if prolonged eye closures suggest the possibility of falling asleep at the wheel.</p> 
   <b>- Looking Away:</b> <p> Notifies the driver if the gaze score indicates diversion of attention from the road.</p> 
   <b>- Distracted:</b> <p> Provides a gentle reminder to refocus attention if the head pose suggests distraction or inattentiveness.</p>
<h3>5. Driver Activity:</h3> 
   <p>-A camera can be placed on the A pillar of the car facing the driver to monitor the activity of the driver.</p>
   <p>-The monitoring system classifies driver activities into 10 distinct categories: <br>
	<b>Safe driving <br>
	Radio operation <br>
	Drinking <br>
	Reaching behind <br>
	Hair and makeup <br>
	Talking to passenger(s) <br>
	Texting with the right hand <br>
	Talking on the phone with the right hand <br>
	Texting with the left hand <br>
	Talking on the phone with the left hand  </b> <br> </p>
   <p>-The dataset used for training and validation is sourced from the State Farm Distracted Driver Detection dataset, available at https://www.kaggle.com/c/state-farm-distracted-driver-detection.</p>
  <p> -Implemented a standard CNN architecture with 4 convolutional layers, 4 max-pooling layers, dropout layers, and flattening layers.</p>
  <p> -The model demonstrated an impressive 97% accuracy in predicting driver behavior. </p> 
<h3>6. Face Recognition:</h3> 
   <b>-Camera Installation:</b> <p>DriveWise incorporates a camera positioned on the speedometer, directly facing the driver, enabling continuous monitoring.</p> 
  <b>-Driver Identification:</b>  <p>Utilizing the KNN algorithm, DriveWise accurately identifies drivers based on facial recognition.</p> 
   <b>-Pre-Registration:</b> <p> Before deployment, employers register all drivers within the system, ensuring accurate identification and eliminating cases where the KNN algorithm classifies unregistered output as one of the classes.</p> 
  <b> -Violation Tracking:</b> <p> If a driver commits a traffic violation, government-installed cameras capture the car's license plate and timestamp, leading to a fine. Subsequently, the employer or owner can identify the driver responsible for the infraction while driving the specific vehicle at that time, facilitating appropriate reprimand. </p>

<h2>Installation and Setup</h2> 
<p>
-Clone the DriveWise repository from GitHub.<br>
-Install required dependencies as listed in the requirements.txt file.<br>
-Configure camera setups for face detection, driver activity monitoring, and face recognition.<br>
-Train the model for driver activity monitoring using the provided dataset.<br>
-Integrate face recognition functionality using the KNN algorithm.<br>
-Customize settings and thresholds as per requirements.</p>

