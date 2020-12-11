#!/usr/env/bin python3

import math
import numpy as np

import pilco
import rospy
import rospkg
import random
import dill
import os.path
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
from sensor_msgs.msg import JointState

rospack = rospkg.RosPack()

MODEL_PATH = os.path.join(rospack.get_path('pilco_control'), 'scripts/multiple_rollout_60t.pkl')
TOPIC = '/double_pendulum_joints_controller/command'

with open(MODEL_PATH, "rb") as f:
    pilco_model = dill.load(f)

pub = rospy.Publisher(TOPIC, Float64MultiArray, queue_size=10)

def data_callback(data):
    global DATAROWS
    datarow = {
        'seq': data.header.seq,
        'position_lower': data.position[0] % (2 * math.pi),
        'position_upper': data.position[1] % (2 * math.pi),
        'velocity_lower': data.velocity[0],
        'velocity_upper': data.velocity[1],
        'effort_lower': data.effort[0],
        'effort_upper': data.effort[1]
    }

    input_vec = np.array([ datarow['position_lower'], datarow['position_upper'], 
        datarow['velocity_lower'], datarow['velocity_upper'] ])
    command(input_vec)


def command(input_vec):
    global pilco_model
    global pub
    action = np.reshape(pilco_model.compute_action(input_vec).numpy(), (-1, 1))
    # print(action)
    val1 = action[1]
    val2 = action[0]

    arg = Float64MultiArray()
    dim_obj = MultiArrayDimension()

    dim_obj.label = ''
    dim_obj.size = 2
    dim_obj.stride = 1

    arg.layout.dim.append(dim_obj)
    arg.data = [val1, val2]
    pub.publish(arg)

def pilco_talker(mode='control', p=10):
    rospy.init_node('pilco_talker')
    print(f"RUNNING PILCO CONTROLLER NODE WITH mode={mode} !")
    if mode=='rand':
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            val1 = (random.random() * 2 * p) - p
            val2 = (random.random() * 2 * p) - p

            arg = Float64MultiArray()
            dim_obj = MultiArrayDimension()

            dim_obj.label = ''
            dim_obj.size = 2
            dim_obj.stride = 1

            arg.layout.dim.append(dim_obj)
            arg.data = [val1, val2]
            pub.publish(arg)
            rate.sleep()
    elif mode == 'control':
        rospy.Subscriber('/joint_states', JointState, data_callback)
        rospy.spin()
    else:
        while not rospy.is_shutdown():
            val1 = 0.0 #(random.random() * 2 * p) - p
            val2 = 0 #(random.random() * 2 * p) - p

            arg = Float64MultiArray()
            dim_obj = MultiArrayDimension()

            dim_obj.label = ''
            dim_obj.size = 2
            dim_obj.stride = 1

            arg.layout.dim.append(dim_obj)
            arg.data = [val1, val2]
            pub.publish(arg)

if __name__ == '__main__':
    mode = rospy.get_param('mode', 'control')
    pilco_talker(mode=mode, p=5.4)
