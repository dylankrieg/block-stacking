# Lab 3 - Oriented Grasping
# Dylan Kriegman

# import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg, mat
from numpy import arctan
import cmath
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt
from math import pi as pi
import math
from controller import Supervisor
from math import cos, sin, atan2, acos, asin, sqrt, pi
from spatialmath import SE3, Twist3
import time

class InverseKinematics():
    def __init__(self):
        self.d1,self.a2,self.a3,self.d4,self.d5,self.d6 = 0.1625,-0.425,-0.3922,0.1333,0.0997,0.0996
        self.d = np.matrix([self.d1, 0, 0, self.d4, self.d5, self.d6])
        self.a = np.matrix([0, self.a2, self.a3, 0, 0, 0])
        self.alph = np.matrix([pi / 2, 0, 0, pi / 2, -pi / 2, 0])

    def AH(self,n, th,c):
        # n: the link
        # th: vector of angles of each joint?
        # d=
        T_a = np.matrix(np.identity(4), copy=False)

        T_a[0, 3] = self.a[0, n - 1]
        T_d = np.matrix(np.identity(4), copy=False)
        T_d[2, 3] = self.d[0, n - 1]

        Rzt = np.matrix([[cos(th[n - 1, c]), -sin(th[n - 1, c]), 0, 0],
                   [sin(th[n - 1, c]), cos(th[n - 1, c]), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], copy=False)

        Rxa = np.matrix([[1, 0, 0, 0],
                   [0, cos(self.alph[0, n - 1]), -sin(self.alph[0, n - 1]), 0],
                   [0, sin(self.alph[0, n - 1]), cos(self.alph[0, n - 1]), 0],
                   [0, 0, 0, 1]], copy=False)

        A_i = T_d * Rzt * T_a * Rxa
        return A_i

    def HTrans(self,th, c):
        # th is a 6 x 8 matrix where each column is a different solution
        # c is the index of the column of theta used
        A_1 = self.AH(1, th, c)
        A_2 = self.AH(2, th, c)
        A_3 = self.AH(3, th, c)
        A_4 = self.AH(4, th, c)
        A_5 = self.AH(5, th, c)
        A_6 = self.AH(6, th, c)
        T_06 = A_1 * A_2 * A_3 * A_4 * A_5 * A_6
        return T_06

    def invKine(self,desired_pos):
        # Returns 6 by 8 matrix where each column is a different solution
        # desired_pos is the homogenuous transform for some (x,y,z) position
        th = np.matrix(np.zeros((6, 8)))
        P_05 = desired_pos * np.matrix([0, 0, -self.d6, 1]).T - np.matrix([0, 0, 0, 1]).T

        # **** theta1 ****
        psi = atan2(P_05[2 - 1, 0], P_05[1 - 1, 0])
        phi = acos(self.d4 / sqrt(P_05[2 - 1, 0] * P_05[2 - 1, 0] + P_05[1 - 1, 0] * P_05[1 - 1, 0]))
        # The two solutions for theta1 correspond to the shoulder
        # being either left or right
        th[0, 0:4] = pi / 2 + psi + phi
        th[0, 4:8] = pi / 2 + psi - phi
        th = th.real
        # **** theta5 ****
        cl = [0, 4]  # wrist up or down
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_16 = T_10 * desired_pos
            th[4, c:c + 2] = + acos((T_16[2, 3] - self.d4) / self.d6)
            th[4, c + 2:c + 4] = - acos((T_16[2, 3] - self.d4) / self.d6)
        th = th.real

        # **** theta6 ****
        # theta6 is not well-defined when sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.
        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_16 = linalg.inv(T_10 * desired_pos)
            th[5, c:c + 2] = atan2((-T_16[1, 2] / sin(th[4, c])), (T_16[0, 2] / sin(th[4, c])))
        th = th.real

        # **** theta3 ****
        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_65 = self.AH(6, th, c)
            T_54 = self.AH(5, th, c)
            T_14 = (T_10 * desired_pos) * linalg.inv(T_54 * T_65)
            P_13 = T_14 * np.matrix([0, -self.d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T
            t3 = cmath.acos((linalg.norm(P_13) ** 2 - self.a2 ** 2 - self.a3 ** 2) / (2 * self.a2 * self.a3))  # norm ?
            th[2, c] = t3.real
            th[2, c + 1] = -t3.real

        # **** theta2 and theta 4 ****
        cl = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_65 = linalg.inv(self.AH(6, th, c))
            T_54 = linalg.inv(self.AH(5, th, c))
            T_14 = (T_10 * desired_pos) * T_65 * T_54
            P_13 = T_14 * np.matrix([0, -self.d4, 0, 1]).T - np.matrix([0, 0, 0, 1]).T

            # theta 2
            th[1, c] = -atan2(P_13[1], -P_13[0]) + asin(self.a3 * sin(th[2, c]) / linalg.norm(P_13))

            # theta 4
            T_32 = linalg.inv(self.AH(3, th, c))
            T_21 = linalg.inv(self.AH(2, th, c))
            T_34 = T_32 * T_21 * T_14
            th[3, c] = atan2(T_34[1, 0], T_34[0, 0])
        th = th.real
        return th


class ArmController():
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        self.camera.recognitionEnable(self.timestep)
        self.camera.enableRecognitionSegmentation()
        self.rangeFinder = self.robot.getDevice("range-finder")
        self.rangeFinder.enable(self.timestep)
        self.plane =  [0, 0, 1, 0.05]
        self.IK = InverseKinematics()
        self.motors = []
        self.armSensors = []
        armMotorNames = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
                         "wrist_3_joint"]
        armMotorSensorNames = ["shoulder_pan_joint_sensor", "shoulder_lift_joint_sensor", "elbow_joint_sensor",
                               "wrist_1_joint_sensor", "wrist_2_joint_sensor",
                               "wrist_3_joint_sensor"]

        self.speed = 1.0
        for i in range(0, len(armMotorNames)):
            self.motors.append(self.robot.getDevice(armMotorNames[i]))
            self.motors[i].setVelocity(self.speed)
            self.armSensors.append(self.robot.getDevice(armMotorSensorNames[i]))
            self.armSensors[i].enable(self.timestep)

        handMotorNames = ["finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1"]
        self.handMotors = []
        for i in range(0, len(handMotorNames)):
            self.handMotors.append(self.robot.getDevice(handMotorNames[i]))
            self.handMotors[i].setVelocity(0.5)

    def moveToHomeState(self):
        print("Moving to Home State")
        homePoseAngles = np.array([[0], [-1.382], [-1.13], [-2.2], [1.63], [0]])
        # homePoseAngles = np.array([[0], [-1.382], [-1.13], [-2.2], [1.63], [pi/3]])
        self.setJointAngles(homePoseAngles)
        for i in range(0, 300):
            self.robot.step(self.timestep)

    # Returns true if a given set of joint angles is above the plane
    def isJointPosSafe(self,joints, plane):
        # joints is a 6 x 1 vector of joint angles
        # plane is a 1 x 4 list of coefficients a,b,c,d
        T_joints = []
        for i in range(0,6):
            T_joints.append(self.IK.AH(i+1, joints, 0))

        # list of h-transforms for each joint
        T = []
        for i in range(0,6):
            T.append(T_joints[0])
            for j in range(1, i + 1):
                T[i] = T[i] * T_joints[j]

        # check if each point is above the plane
        for i in range(0,len(T)):
            jointCoordinate = T[i][0:,3].flatten().tolist()[0]
            # check if point is above the plane (ax + by + cz + d = 0 ))
            # print((plane[0]*jointCoordinate[0]) + (plane[1]*jointCoordinate[1]) + (plane[2] * jointCoordinate[2]))
            if (plane[0]*jointCoordinate[0]) + (plane[1]*jointCoordinate[1]) + (plane[2] * jointCoordinate[2]) < plane[3]:
                return False
        return True

    # returns a 6 x 1 Numpy Array of joint angles
    def getJointPositions(self):
        thetas = np.zeros((6, 1))
        for j in range(0, 6):
            thetas[j, 0] = self.armSensors[j].getValue()
        return thetas

    def setJointAngles(self,thetas):
        # thetas is a 6 x 1 vector of joint angles
        for i in range(0, 6):
            theta_i = thetas[i]
            self.motors[i].setPosition(float(theta_i))

    def moveJ(self,T_06):
        # T_06 is the 4 x 4 homogenuous transform for position and rotation from the grippers coordinate frame to the base's coordinate frame

        jointAngleSolutions = self.IK.invKine(T_06)

        # Reject all joint positions that lead to an unsafe position
        # Uses the first safe set of angles
        solutionFound = False
        for col in range(0, 8):
            sol = jointAngleSolutions[0:6, col]
            if self.isJointPosSafe(sol, self.plane):
                solutionFound = True
                break

        # if there are no safe configuration
        if solutionFound == False:
            raise Exception(f"No safe joint configuration for pose {T_06}")

        # Use the maximum angle that the robot has to move to compute the number of timesteps required to perform that motion
        jointPositions = self.getJointPositions()
        maxAngleMoved = np.max(np.abs(jointPositions - sol))
        maxAngularVelocity = math.pi / 2  # maximum angular velocity of the wrist joints (180 degrees/sec)
        timeRequired = maxAngleMoved / maxAngularVelocity
        timeStepsRequired = int(timeRequired / (self.timestep / 1000)) + 1
        # Discretize the joint angles into the appropriate number of bins
        theta0_range = np.linspace(jointPositions[0, 0], sol[0, 0], timeStepsRequired)
        theta1_range = np.linspace(jointPositions[1, 0], sol[1, 0], timeStepsRequired)
        theta2_range = np.linspace(jointPositions[2, 0], sol[2, 0], timeStepsRequired)
        theta3_range = np.linspace(jointPositions[3, 0], sol[3, 0], timeStepsRequired)
        theta4_range = np.linspace(jointPositions[4, 0], sol[4, 0], timeStepsRequired)
        theta5_range = np.linspace(jointPositions[5, 0], sol[5, 0], timeStepsRequired)

        # Incrementally move the joints there
        for i in range(0, timeStepsRequired):
            thetas = np.mat([theta0_range[i], theta1_range[i], theta2_range[i], theta3_range[i], theta4_range[i],
                             theta5_range[i]]).T
            self.setJointAngles(thetas)
            self.robot.step(self.timestep)
            self.robot.step(self.timestep)

    def moveL(self,T_06):
        # T_06 is the 4 x 4 homogenuous transform for position and rotation from the grippers coordinate frame located at the goal to the base's coordinate frame
        # Use the forward kinematics to calculate the current position of the robot
        T_06_init = self.getPose()
        # initOrientation = T_06_init[0:3, 0:3]
        goalOrientation = T_06[0:3, 0:3]
        gripperFrameOrigin = T_06_init[0:4, 3]
        x, y, z = gripperFrameOrigin.item((0,0)), gripperFrameOrigin.item((1,0)), gripperFrameOrigin.item((2,0))
        # Determine the Euclidian distance between the current and the desired position of the robot
        goalX, goalY, goalZ = T_06.item((0, 3)), T_06.item((1, 3)), T_06.item((2, 3))
        dist = linalg.norm(np.array([goalX,goalY,goalZ])-np.array([x,y,z])) # distance in meters

        # Assume a maximum Cartesian speed of the robot, e.g. 1cm/s, and generate a list of waypoints from the current to the desired position of the robot
        # fix the orientation of the robot and vary the position
        maxSpeed = 100  # in cm/s
        timeRequired = (dist * 100) / maxSpeed
        timeStepsRequired = int(timeRequired / (self.timestep / 1000)) + 1
        xRange = np.linspace(x, goalX, timeStepsRequired)
        yRange = np.linspace(y, goalY, timeStepsRequired)
        zRange = np.linspace(z, goalZ, timeStepsRequired)
        wayPointTransforms = []
        for i in range(1, len(xRange)):
            goalX, goalY, goalZ = xRange[i], yRange[i], zRange[i]
            # fix the pose of the gripper but vary the position of it's center
            wayPointTransform = np.matrix(np.zeros((4, 4)))
            wayPointTransform[3, 3] = 1
            wayPointTransform[0:3, 0:3] = goalOrientation
            wayPointTransform[0, 3] = goalX
            wayPointTransform[1, 3] = goalY
            wayPointTransform[2, 3] = goalZ
            wayPointTransforms.append(wayPointTransform)
            print(i)

        # Create a control loop that calls MoveJ to move from point to point. For smoother motion, you can pull the next waypoint, once the robot gets reasonably close to the next waypoint.
        for transform in wayPointTransforms:
            # move to pose given by transform
            self.moveJ(transform)

    def closeGripper(self):
        print("Closing Gripper")
        for i in range(0, len(self.handMotors)):
            self.handMotors[i].setPosition(1)
        for i in range(0, 500):
            self.robot.step(self.timestep)

    def openGripper(self):
        print("Opening Gripper")
        for i in range(0, len(self.handMotors)):
            self.handMotors[i].setPosition(self.handMotors[i].getMinPosition())
        for i in range(0, 500):
            self.robot.step(self.timestep)

    def getPose(self):
        # returns the coordinate transform in the robot's frame associated with the current pose
        jointPositions = self.getJointPositions()
        T_06 = self.IK.HTrans(jointPositions,0)
        return np.matrix(T_06)

    def getGoalPose(self,dX,dY,dZ):
        # returns a new homogenous transorm with x,y,z incremented by dX,dY,dZ
        currentPose = self.getPose()
        print(f"Current Pose: \n {currentPose}")
        goalPose = np.array(currentPose)
        print(f"dX: {dX}")
        print(f"dY: {dY}")
        print(f"dZ: {dZ}")
        goalPose[0,3] += dX
        goalPose[1,3] += dY
        goalPose[2,3] += dZ
        print(f"Goal Pose: {goalPose}")
        return goalPose

    def getCameraImage(self):
        imageWidth,imageHeight = self.camera.getWidth(),self.camera.getHeight()
        image1D = self.camera.getImage()
        image = np.frombuffer(image1D, np.uint8).reshape((imageHeight, imageWidth, 4))
        return image

    def getSegmentedImage(self):
        # returns segemented image as np array with shape (320,240,4)
        imageWidth,imageHeight = self.camera.getWidth(),self.camera.getHeight()
        image1DSeg = self.camera.getRecognitionSegmentationImage()
        imageSeg = np.frombuffer(image1DSeg, np.uint8).reshape((imageHeight, imageWidth, 4))
        return imageSeg

    def getDepthImage(self):
        depth_1darray = np.frombuffer(self.rangeFinder.getRangeImage(data_type="buffer"), dtype=np.float32)
        depthImage = np.reshape(depth_1darray,(240,320))
        depthImage = depthImage * 1000
        return depthImage

    def collectSampleData(self):
        cameraImage = self.getCameraImage()
        segImage = self.getSegmentedImage()
        depthImage = self.getDepthImage()
        plt.imshow(cameraImage)
        plt.show()
        plt.imshow(depthImage)
        plt.show()
        plt.imshow(segImage)
        plt.show()
        np.save("depth_image_4", depthImage)
        np.save("seg_image_4", segImage)
        np.save("reg_image_4", cameraImage)


    def runRoutine(self):
        configuration_1  = np.array([0,math.radians(-45),math.radians(90),math.radians(-45),math.radians(90),0])
        configuration_2  = np.array([0,math.radians(45),math.radians(-90),math.radians(-45),math.radians(90),0])

        self.wait(0.01)
        print("Setting joint angles")
        self.moveL(self.getGoalPose(-0.8,0.5,0))
        self.wait(4)
        '''
        # print("Moving to home configuration")
        gripperFrameOrigin = self.getPose()[0:4, 3]
        print(f"gripperFrameOrigin:\n{gripperFrameOrigin}")
        x, y, z = gripperFrameOrigin.item((0, 0)), gripperFrameOrigin.item((1, 0)), gripperFrameOrigin.item((2, 0))
        robotCoordsDuringImage = np.array((x,y,z))
        # move down 0.0246 to reach height of red cube
        redRobotCoordinates = np.array((-0.22738437727093697, 0.024643749743700025, 0.5820000171661377))
        dX = -0.5580000281333923
        dY = 0.22250000713393092
        dZ = 0.10611458234488964

        # dX,dY,dZ = (robotCoordsDuringImage - redRobotCoordinates)
        # print(dZ)
        goalPose = self.getGoalPose(dX,dY,dZ)
        print(f"goalPose {goalPose}")
        self.moveL(goalPose)
        print("Moved to goal pose")
        '''
        self.wait(1)

    def wait(self,seconds):
        for i in range(1,int(seconds*1000),self.timestep):
            self.robot.step(self.timestep)


armController = ArmController()
armController.runRoutine()



'''
def searchForCan():
    print("Searching for can...")
    global state,goalPose

    # Take depth image
    depth1D = np.frombuffer(rangefinder.getRangeImage(data_type="buffer"), dtype=np.float32)
    print(depth1D.shape)
    depthImage = np.reshape(depth1D, (240, 320))
    depthImage = depthImage * 1000.0

    imageWidth, imageHeight = camera.getWidth(), camera.getHeight()
    image1DSeg = camera.getRecognitionSegmentationImage()
    imageSeg = np.frombuffer(image1DSeg, np.uint8).reshape((imageHeight,imageWidth, 4))

    imageSegMono = np.dot(imageSeg,[0.2989, 0.5870, 0.1140, 0]).astype(int)
    robot.step(timestep)

    # mask the depth image using the monochromatic segmented image so that only cans have defined values
    depthImageSeg = np.multiply(depthImage, imageSegMono == 254)

    rgbdImageSeg = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(imageSeg),
        o3d.geometry.Image(np.array(depthImageSeg).astype('uint16')),
        convert_rgb_to_intensity=True,
        depth_scale=1000.0, depth_trunc=1.25)

    canPCD = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbdImageSeg,
        o3d.camera.PinholeCameraIntrinsic(320, 240, 320, 240, 160, 120),
        project_valid_depth_only=True
    )
    #canPCD.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    print(f"Current Gripper Frame: \n{getPose()}")
    print(f"Current Gripper Coords in World: \n{getPose()[0:3,3]}")

    canOBB = canPCD.get_oriented_bounding_box()
    canOBB.color = (0,1,0)

    # The can's center (x,y,z) in the camera's coordinate frame from PCA
    xCam, yCam= canOBB.get_center()[0:2]
    zCam = canOBB.get_min_bound()[2]
    #xCam,yCam,zCam = canOBB.get_center()[0:3]
    print(f"Can Coords (Camera Frame): \n{[xCam,yCam,zCam]}")

    # Rotation from camera frame to gripper frame (None)
    R_GC = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # Points location in the camera frame
    P_C = np.matrix([[xCam],[yCam],[zCam]])

    gripperWidth = 0.05
    # Vector from the center of the last joint to the center of the gripper plate
    # Requires more transformations... so just doing it simply
    # offsetVector = gripperWidth * np.matrix(getPose()[0:3,2])
    offsetVector = np.matrix([[0],[0],[gripperWidth + 0.01]])

    # Origin of cameras frame relative to gripper frame
    O_CG = offsetVector
    print(O_CG)
    print(O_CG.shape)
    # Points location in gripper frame
    P_G = np.matmul(R_GC,P_C) + O_CG
    print(f"Can Coords (Gripper Frame): \n{P_G}")

    # WTF points location in the world frame
    # Trasformation of gripper frame in world frame
    T_GW = getPose()

    # Rotation from gripper frame to world frame
    R_WG = T_GW[0:3,0:3]

    # Origin of grippers frame relative to world frame
    O_WG = T_GW[0:3,3]

    # Points location in the world frame
    P_W = np.matmul(R_WG,P_G) + O_WG
    print(f"Can Coords (World): \n{P_W}")

    # goalPose is 4x4 Homogenous Coordinates of Gripper relative to the World Frame
    goalPose = np.matrix(np.zeros((4,4)))
    goalPose[3,3] = 1

    # Move the gripper to P_W
    goalPose[0:3,3] = P_W

    # Rotation Code
    # WTF point that describes unit vector for x-axis (longest axis) of can in world coordinates
    # Rotation from camera frame to gripper frame
    R_CanC = canOBB.R

    # WTF point in the world frame
    # Points location in the camera frame
    P_C = np.matrix([[R_CanC[0][0]], [R_CanC[1][0]], [R_CanC[2][0]]])
    print(f"X-axis vector in camera frame: \n {P_C}")

    # Points location in gripper frame
    R_GC = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    P_G = np.matmul(R_GC, P_C) + O_CG
    print(f"X-axis vector in gripper frame: \n {P_G}")

    # Can's x-axis vector in the world frame
    P_W = np.matmul(R_WG, P_G) + O_WG

    # Make the magnitude 1 st. P_W is a unit vector
    P_W[2,0] = 0
    P_W = P_W / np.linalg.norm(P_W)
    print(f"X-axis vector in world frame (projected on xy plane):\n {P_W}")

    # Align y-axis of gripper to x-axis (longest axis from PCA) of can in plane
    # Align z-axis vertical
    # Align x-axis perpindicular to y-axis (by properties)
    goalPoseRotation = np.mat([[-P_W[1][0],P_W[0][0],0],[P_W[0][0],P_W[1][0],0],[0,0,-1]])

    # Set the rotation in the goal pose
    goalPose[0:3,0:3] = goalPoseRotation

    # Move the gripper to goalPose and close the gripper
    state = "pickUpCan"
'''

