https://github.com/tensorflow/models/tree/master/research/object_detection

(Using Faster Rcnn with Inception, tensorflow 1.14)

Follow the instructions to install the tensorflow object detection api

Copy contents of this vision_task dir to tensorflow api.

Input images should be saved in directory named: productionpanos

Output images will be saved in directory named: productionpanos_out

Each output image will remaned according to the object it contains (classification for goal 1)
and will draw bounding box of the detected region as well (localization for goal 2)
