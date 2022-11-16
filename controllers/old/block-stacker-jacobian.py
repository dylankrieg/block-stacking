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
        self.d = np.array([self.d1, 0, 0, self.d4, self.d5, self.d6])
        self.a = np.array([0, self.a2, self.a3, 0, 0, 0])
        self.alph = np.array([pi / 2, 0, 0, pi / 2, -pi / 2, 0])
        self.zero_config_fk = SE3(self.HTrans([0]*6)[-1])

    def AH(self,n, th):
        # n: the link
        # th: vector of angles of each joint?
        T_a = np.array(np.identity(4), copy=False)
        T_a[0, 3] = self.a[n - 1]
        T_d = np.array(np.identity(4), copy=False)
        T_d[2, 3] = self.d[n - 1]

        Rzt = np.array(
            [
                [cos(th[n - 1]), -sin(th[n - 1]), 0, 0],
                [sin(th[n - 1]), cos(th[n - 1]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            copy=False
        )
        Rxa = np.array(
            [
                [1, 0, 0, 0],
                [0, cos(self.alph[n - 1]), -sin(self.alph[n - 1]), 0],
                [0, sin(self.alph[n - 1]), cos(self.alph[n - 1]), 0],
                [0, 0, 0, 1],
            ],
            copy=False
        )

        A_i = T_d @ Rzt @ T_a @ Rxa

        return A_i

    def HTrans(self,th):
        # th is a 6 x 8 matrix where each column is a different solution
        A_1 = self.AH(1, th)
        A_2 = self.AH(2, th)
        A_3 = self.AH(3, th)
        A_4 = self.AH(4, th)
        A_5 = self.AH(5, th)
        A_6 = self.AH(6, th)

        T_01 = A_1
        T_02 = T_01 @ A_2
        T_03 = T_02 @ A_3
        T_04 = T_03 @ A_4
        T_05 = T_04 @ A_5
        T_06 = T_05 @ A_6

        transforms = [T_01, T_02, T_03, T_04, T_05, T_06]
        return transforms

    def get_joint_twists(self):
        # everything in the space frame aka base frame
        joint_twists = []

        # first joint
        axis = np.array([0, 0, 1])  # rotates around z, right hand rule
        point = np.array([0, 0, 0])  # a point on the axis of rotation
        twist = Twist3.UnitRevolute(axis, point)
        joint_twists.append(twist)

        # second joint
        axis = np.array([0, -1, 0])
        point = np.array([0, 0, self.d1])
        twist = Twist3.UnitRevolute(axis, point)
        joint_twists.append(twist)

        # third joint
        axis = np.array([0, -1, 0])
        point = np.array([self.a2, 0, self.d1])
        twist = Twist3.UnitRevolute(axis, point)
        joint_twists.append(twist)

        # fourth joint
        axis = np.array([0, -1, 0])
        point = np.array([self.a2 + self.a3, -self.d4, self.d1])
        twist = Twist3.UnitRevolute(axis, point)
        joint_twists.append(twist)

        # fifth joint
        axis = np.array([0, 0, -1])
        point = np.array([self.a2 + self.a3, -self.d4, self.d1 - self.d5])
        twist = Twist3.UnitRevolute(axis, point)
        joint_twists.append(twist)

        # sixth joint
        axis = np.array([0, -1, 0])
        point = np.array([self.a2 + self.a3, -self.d4 - self.d6, self.d1 - self.d5])
        twist = Twist3.UnitRevolute(axis, point)
        joint_twists.append(twist)

        return joint_twists

    def get_fk_from_twists(self,joint_angles):
        joint_twists = self.get_joint_twists()
        relative_transforms = []
        for idx, joint_twist in enumerate(joint_twists):
            angle = joint_angles[idx]
            transform = SE3(joint_twist.exp(angle))
            relative_transforms.append(transform)

        fk = self.zero_config_fk
        for transform in relative_transforms[::-1]:  # apply in reverse order
            fk = transform * fk
        return fk

    def get_ur5e_jacobian_from_twists(self,angles, frame=None):
        if frame is None:
            frame = "body"
        joint_twists = self.get_joint_twists()
        relative_transforms = []
        for idx, joint_twist in enumerate(joint_twists):
            angle = angles[idx]
            print(angle)
            print(joint_twist.exp(angle))
            relative_transforms.append(SE3(joint_twist.exp(angle)))
        jacobian = np.zeros([6, 6])
        twist_transform = SE3(np.eye(4))
        for idx in range(6):
            if idx > 0:
                twist_transform = twist_transform @ relative_transforms[idx - 1]
            jacobian[:, idx] = twist_transform.Ad() @ joint_twists[idx].A

        if frame == "space":
            return jacobian
        elif frame == "body":
            fk = self.zero_config_fk
            for transform in relative_transforms[::-1]:  # apply in reverse order
                fk = transform * fk
            return fk.inv().Ad() @ jacobian
        else:
            raise Exception(f"frame: {frame} not in (space, body)")



    def get_adjoint(self,angles):
        current_transform = self.get_fk_from_twists(angles).A
        adjoint = SE3(current_transform).Ad()
        return adjoint

    def get_adjoint_inverse(self,angles):
        current_transform = self.get_fk_from_twists(angles).A
        adjoint_inverse = SE3(current_transform).inv().Ad()
        return adjoint_inverse

    def get_body_twist_from_transform(self,desired_transform, current_transform):
        """
        Even though both desired_transform and current_transform are in space frame,
        this returns a twist in the body frame.
        """
        transform_from_desired = SE3(current_transform).inv().A @ desired_transform
        twist = SE3(transform_from_desired).log(twist=True)
        return twist

    def get_body_twist(self,angles, desired_transform):
        transforms = self.HTrans(angles)
        current_transform = transforms[-1]
        body_twist = self.get_body_twist_from_transform(desired_transform, current_transform)
        return body_twist

    def get_space_twist(self,angles, desired_transform):
        body_twist = self.get_body_twist(angles, desired_transform)
        space_twist = self.get_adjoint(angles) @ body_twist
        return space_twist

    def get_twist(self,angles, desired_transform, frame=None):
        if frame is None or frame == "body":
            return self.get_body_twist(angles, desired_transform)
        elif frame == "space":
            return self.get_space_twist(angles, desired_transform)
        else:
            raise Exception(f"frame: {frame} not in (space, body)")

    def damped_pinv(self,J, rho=1e-4):
        assert J.shape == (6, 6)  # for UR5e, remove otherwise
        rho_squared = rho * rho
        output = J.T @ np.linalg.pinv(J @ J.T + rho_squared * np.eye(J.shape[0]))
        return output

    def damped_scaled_pinv(self,J, rho=1e-3):
        assert J.shape == (6, 6)  # for UR5e, remove otherwise
        rho_squared = rho * rho
        jjt = J @ J.T
        diag_j = np.diag(np.diag(jjt))  # call np.diag twice, first to get diagonal, second to reshape
        output = J.T @ np.linalg.pinv(jjt + rho_squared * diag_j)
        return output

    def get_trajectory(self,target,joint_angles=None,pinv_func=None,debug=False,max_iter=100,learning_rate=0.1):
        if joint_angles is None:
            joint_angles = [0, 0, 0, 0, 0, 0]

        if pinv_func is None:
            pinv_func = np.linalg.pinv

        epsilon_v = 1e-4
        epsilon_w = 1e-4
        output = [joint_angles]

        joint_angles = np.array(joint_angles)
        FRAME = "space"
        #     FRAME = "body"
        J = self.get_ur5e_jacobian_from_twists(joint_angles, frame=FRAME)
        J_pinv = pinv_func(J)
        twist = self.get_twist(joint_angles, target, frame=FRAME)
        twist[np.isnan(twist)] = 0

        count = 0
        norm = np.linalg.norm
        while (count < max_iter and (norm(twist[:3]) > epsilon_v or norm(twist[3:]) > epsilon_w)):
            step = J_pinv @ twist
            if debug:
                print(f"step: {step.round(3)}")
            joint_angles = joint_angles + learning_rate * step
            if debug:
                print(self.HTrans(joint_angles)[-1].round(3))

            J = self.get_ur5e_jacobian_from_twists(joint_angles, frame=FRAME)
            J_pinv = pinv_func(J)
            twist = self.get_twist(joint_angles, target, frame=FRAME)
            twist[np.isnan(twist)] = 0.
            if debug:
                print(f"twist: {twist.round(3)}")
            output.append(joint_angles)
            count += 1
        return output, twist

    def moveTo(self,target,joint_angles,ArmController):
        """
        move to the target homogeneous transform from current joint_angles
        """
        output, error = self.get_trajectory(target,joint_angles,pinv_func=self.damped_scaled_pinv)

        last_config = output[-1]
        ArmController.setJointAngles(last_config)
        for i in range(100):
            ArmController.robot.step(32)
        return output


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
        self.plane =  [0, 0, 1, 0.01]
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
        for i in range(0, 100):
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
            self.motors[i].setPosition(theta_i)

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

    '''

    def getPose(self):
        # returns the coordinate transform in the robot's frame associated with the current pose
        jointPositions = self.getJointPositions()
        T_06 = self.IK.HTrans(jointPositions)
        print(T_06)
        return np.array(T_06)

    def getGoalPose(self,dX,dY,dZ):
        # returns a new homogenous transorm with x,y,z incremented by dX,dY,dZ
        currentPose = self.getPose()
        print(currentPose)
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
    '''

    def moveJ(self,T_06):
        # T_06 is the 4 x 4 homogenuous transform for position and rotation from the grippers coordinate frame to the base's coordinate frame

        jointAngleSolutions = self.IK.invKine(T_06)

        # Reject all joint positions that lead to an unsafe position
        # Uses the first safe set of angles
        solutionFound = False
        for col in range(0, 8):
            sol = jointAngleSolutions[0:6, col]
            if self.isJointPosSafe(sol,self.plane):
                solutionFound = True
                break

        # if there are no safe configuration
        if solutionFound == False:
            raise Exception("No safe joint configuration for pose")

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
            thetas = np.mat([theta0_range[i], theta1_range[i], theta2_range[i], theta3_range[i], theta4_range[i], theta5_range[i]]).T
            self.setJointAngles(thetas)
            self.robot.step(self.timestep)
            self.robot.step(self.timestep)

    def moveL(self,T_06):
        # T_06 is the 4 x 4 homogenuous transform for position and rotation from the grippers coordinate frame located at the goal to the base's coordinate frame
        # Use the forward kinematics to calculate the current position of the robot
        T_06_init = self.getPose()
        # initOrientation = T_06_init[0:3, 0:3]
        goalOrientation = T_06[0:3,0:3]
        gripperFrameOrigin = T_06_init[0:4, 3]
        x, y, z = gripperFrameOrigin[0, 0], gripperFrameOrigin[1, 0], gripperFrameOrigin[2, 0]

        # Determine the Euclidian distance between the current and the desired position of the robot
        goalX, goalY, goalZ = T_06.item((0,3)),T_06.item((1,3)),T_06.item((2,3))
        diff = np.mat([goalX - x, goalY - y, goalZ - z]).T
        dist = np.sqrt(np.sum(np.dot(diff, diff.T)))  # distance in meters

        # Assume a maximum Cartesian speed of the robot, e.g. 1cm/s, and generate a list of waypoints from the current to the desired position of the robot
        # fix the orientation of the robot and vary the position
        maxSpeed = 100 # in cm/s
        timeRequired = (dist * 100) / maxSpeed
        timeStepsRequired = int(timeRequired / (self.timestep / 1000)) + 1
        xRange, yRange, zRange = np.linspace(x, goalX, timeStepsRequired), np.linspace(y, goalY, timeStepsRequired), np.linspace(z,goalZ,timeStepsRequired)
        wayPointTransforms = []
        for i in range(1, len(xRange)):
            goalX, goalY, goalZ = xRange[i],yRange[i],zRange[i]
            # fix the pose of the gripper but vary the position of it's center
            wayPointTransform = np.matrix(np.zeros((4, 4)))
            wayPointTransform[3,3] = 1
            wayPointTransform[0:3, 0:3] = goalOrientation
            wayPointTransform[0, 3] = goalX
            wayPointTransform[1, 3] = goalY
            wayPointTransform[2, 3] = goalZ
            wayPointTransforms.append(wayPointTransform)

        # Create a control loop that calls MoveJ to move from point to point. For smoother motion, you can pull the next waypoint, once the robot gets reasonably close to the next waypoint.
        for transform in wayPointTransforms:
            # move to pose given by transform
            self.moveJ(transform)

    def moveDeltas(self,dX,dY,dZ):
        # goalPose = self.getGoalPose(dX,dY,dZ)
        # print(goalPose)
        x,y,z = 0,0.5,0.25
        goalPose = np.array([[1,0,0,x],
                               [0,1,0,y],
                               [0,0,1,z],
                               [0,0,0,1]])

        # x = self.IK.HTrans(self.getJointPositions())
        newPositions = self.getJointPositions()
        '''
        for xi in x:
            print(xi)
        '''
        # self.IK.moveTo(goalPose,self.getJointPositions(),self)

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
        self.wait(0.01)
        print("Setting joint angles")
        self.setJointAngles(configuration_1)
        self.wait(2)
        # move robot 0.1 units in +x

        # move robot 0.1 units in -x
        # move to home

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


'''
controller.openGripper()
for i in range(1,200):
    controller.robot.step(controller.timestep)
controller.closeGripper()
'''
# robot = controller.robot


# marker_node = robot.getFromDef("Marker")
# marker_trans = marker_node.getField("translation")
# marker_rot  = marker_node.getField("rotation")

# rangefinder = robot.getDevice('range-finder')
# rangefinder.enable(controller.timestep)

# camera = robot.getDevice('camera')
# camera.enable(controller.timestep)
# camera.recognitionEnable(controller.timestep)
# camera.enableRecognitionSegmentation()


# depth1D = np.frombuffer(rangefinder.getRangeImage(data_type="buffer"), dtype=np.float32)
# depthImage = np.reshape(depth1D, (240, 320))
# depthImage = depthImage * 1000.0

# imageWidth, imageHeight = camera.getWidth(), camera.getHeight()
# plt.imshow(depthImage)
# plt.show()
# image1DSeg = camera.get()

# image1DSeg = camera.getRecognitionSegmentationImage()
# print(image1DSeg)
# imageSeg = np.frombuffer(image1DSeg, np.uint8).reshape((imageHeight,imageWidth, 4))

# imageSegMono = np.dot(imageSeg,[0.2989, 0.5870, 0.1140, 0]).astype(int)

# mask the depth image using the monochromatic segmented image so that only cans have defined values
# depthImageSeg = np.multiply(depthImage, imageSegMono == 254)
'''

plt.subplot(1, 3, 1)
plt.title('Camera image')
plt.imshow(imageSeg)
plt.subplot(1, 3, 2)
plt.title('Depth image')
plt.imshow(depthImage)
plt.subplot(1, 3, 3)
plt.title('Monochrome image')
plt.imshow(imageSegMono)
plt.show()

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
'''