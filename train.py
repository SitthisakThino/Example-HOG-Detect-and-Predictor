import os
import sys
import glob

import dlib

options = dlib.simple_object_detector_training_options()

options.add_left_right_image_flips = True

options.C = 5

options.num_threads = 4
options.be_verbose = True

training_xml_path = os.path.join('for train', "train01NShape.xml")
testing_xml_path = os.path.join('for validate', "trainValid.xml")

dlib.train_simple_object_detector(training_xml_path, "detector.svm", options)

print("")  # Print blank line to create gap from previous output
print("Training accuracy: {}".format(
    dlib.test_simple_object_detector(training_xml_path, "detector.svm")))
    
print("Testing accuracy: {}".format(
    dlib.test_simple_object_detector(testing_xml_path, "detector.svm")))