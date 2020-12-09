#!/usr/env/bin python3

import math

import pilco
import rospy
import random
from sensor_msgs.msg import JointState
from glob import glob
import pandas as pd

DATAFRAME = pd.DataFrame(columns=['seq', 'position_upper', 'position_lower', 'velocity_upper', 'velocity_lower', 'effort_upper', 'effort_lower'])

def callback(data):
    global DATAFRAME
    datarow = {
        'seq': data.header.seq,
        'position_lower': (data.position[0] % (2 * math.pi)),
        'position_upper': (data.position[1] % (2 * math.pi)),
        'velocity_lower': data.velocity[0],
        'velocity_upper': data.velocity[1],
        'effort_lower': data.effort[0],
        'effort_upper': data.effort[1]
    }
    print(datarow)
    DATAFRAME = DATAFRAME.append(datarow, ignore_index=True)

def pilco_listener():

    rospy.init_node('pilco_listener', anonymous=True)

    rospy.Subscriber('/joint_states', JointState, callback)

    rospy.spin()

    n = sum([1 for i in glob('./*.csv')])
    DATAFRAME.to_csv(f'pilco_train_data30hz_{n+1}.csv', index=False)

if __name__ == '__main__':
    pilco_listener()

