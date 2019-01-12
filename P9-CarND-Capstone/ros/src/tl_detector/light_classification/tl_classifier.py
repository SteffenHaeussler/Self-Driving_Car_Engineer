from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import yaml
import rospy
import datetime

class TLClassifier(object):
    def __init__(self):

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.safe_load(config_string)
        model_name = 'frozen_inference_graph.pb'

        if self.config['is_site']:
            model_path = 'light_classification/model/real_frozen_inference_graph.pb'
        else:
            model_path = 'light_classification/model/sim_frozen_inference_graph.pb'

        self.frozen_graph = tf.Graph()
        self.threshold = .5

        with self.frozen_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                od_graph_def.ParseFromString(fid.read())
                tf.import_graph_def(od_graph_def, name='')

            self.image_tensor = self.frozen_graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.frozen_graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.frozen_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.frozen_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.frozen_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.frozen_graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: traffic light color ID (specified in styx_msgs/TrafficLight)
        """
        with self.frozen_graph.as_default():

            input_image = np.expand_dims(image, axis=0)

            start = datetime.datetime.now()
            predict = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections],
                                         feed_dict={self.image_tensor: input_image})

            end = datetime.datetime.now()
            diff = end - start
#             print(diff.total_seconds())

        probility = np.squeeze(predict[1])
        predicted_classes = np.squeeze(predict[2]).astype(np.int32)

        if probility[0] > self.threshold:
            print('Prob: ', probility[0])
            print('Class: ', predicted_classes[0])

        if probility[0] > self.threshold:
            if predicted_classes[0] == 1:
                return TrafficLight.GREEN

            elif predicted_classes[0] == 2:
                return TrafficLight.RED

            elif predicted_classes[0] == 3:
                return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
