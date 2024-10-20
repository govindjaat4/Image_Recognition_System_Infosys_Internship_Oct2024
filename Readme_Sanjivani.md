<h1>Face Recognition using MobileFaceNet and MobileNetV2</h1>

<p>This project explores two approaches for face recognition using different models. The initial part involves training the MobileFaceNet model from scratch on the LFW (Labeled Faces in the Wild) dataset available in scikit-learn. However, this experiment yielded suboptimal results, and the model requires further tuning. In the second part, accessing the local machine camera through Google Colab is explored. It contains code snippets to access the camera, take and save snapshots, and stream video continuously in real time. It also includes code for capturing and saving faces in the custom dataset. In the last part, a pre-trained MobileNetV2 model is used for real-time face recognition on a custom dataset, which was created from captured video frames.</p>

<p>Currently, the code provides a foundation for face recognition tasks and is still under development. The camera functionality includes face detection, and the recognition system is designed to perform on-the-fly predictions using MobileNetV2.</p>

<h2>Features</h2>
<ul>
    <li><strong>MobileFaceNet Training</strong>: Custom training of the MobileFaceNet model from scratch on the LFW dataset.</li>
    <li><strong>Camera Access in Google Colab</strong>: Capturing snapshots and continuous video streams using Google Colab's camera access capabilities.</li>
    <li><strong>Real-time Face Recognition</strong>: Using a pre-trained MobileNetV2 model for real-time face recognition on a custom dataset captured from the camera.</li>
    <li><strong>Face Detection</strong>: Basic face detection functionality implemented during video stream capture.</li>
</ul>

<h2>Future Work</h2>
<p>The project is a work in progress. Improvements will be made to enhance the accuracy of the MobileFaceNet model and MobileNetV2. The goal is to improve the recognition process and the confidence level of predictions.</p>

<p>The <strong>imgrec.ipynb</strong> file contains the code for all the mentioned functionalities.</p>
