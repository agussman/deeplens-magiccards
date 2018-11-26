# ----------------------------------- 
# Copyright Amazon AWS DeepLens, 2017
# -----------------------------------

import os
import greengrasssdk
from threading import Timer
import time
import awscam
import cv2
from threading import Thread

# Create an AWS Greengrass core SDK client.
client = greengrasssdk.client('iot-data')

# The information exchanged between AWS IoT and the AWS Cloud has 
# a topic and a message body.
# This is the topic that this code uses to send messages to the Cloud.
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
ret, frame = awscam.getLastFrame()
ret,jpeg = cv2.imencode('.jpg', frame) 
Write_To_FIFO = True
class FIFO_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)
 
    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path,'w')
        client.publish(topic=iotTopic, payload="Opened Pipe")
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue  

def greengrass_infinite_infer_run():
    try:
        input_width = 300
        input_height = 300
        max_threshold = 0.25

        results_thread = FIFO_Thread()
        results_thread.start()
        
        # Send a starting message to the AWS IoT console.
        client.publish(topic=iotTopic, payload="About to do nothing")

        ret, frame = awscam.getLastFrame()

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        global jpeg
        ret,jpeg = cv2.imencode('.jpg', gray_frame)
            
    except Exception as e:
        msg = "Test failed: " + str(e)
        client.publish(topic=iotTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds.
    Timer(15, greengrass_infinite_infer_run).start()

# Execute the function.
greengrass_infinite_infer_run()

# This is a dummy handler and will not be invoked.
# Instead, the code is executed in an infinite loop for our example.
def function_handler(event, context):
    return