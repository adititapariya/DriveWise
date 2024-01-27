import numpy as np
import cv2


def resize(frame, scale_percent):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_LINEAR)
    return resized


def get_face_area(face):
    return abs((face.left() - face.right()) * (face.bottom() - face.top()))


def show_keypoints(keypoints, frame):
    for n in range(0, 68):
        x = keypoints.part(n).x
        y = keypoints.part(n).y
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        return frame


def midpoint(p1, p2):
    return np.array([int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)])


def get_array_keypoints(landmarks, dtype="int", verbose: bool = False):
    points_array = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        points_array[i] = (landmarks.part(i).x, landmarks.part(i).y)

    if verbose:
        print(points_array)

    return points_array


def isRotationMatrix(R, precision=1e-4):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < precision


def rotationMatrixToEulerAngles(R, precision=1e-4):
    # Calculates Tait–Bryan Euler angles from a Rotation Matrix
    assert (isRotationMatrix(R, precision))  # check if it's a Rmat

    # assert that sqrt(R11^2 + R21^2) != 0
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < precision

    if not singular:  # if not in a singularity, use the standard formula
        x = np.arctan2(R[2, 1], R[2, 2])  # atan2(R31, R33) -> YAW, angle PSI

        # atan2(-R31, sqrt(R11^2 + R21^2)) -> PITCH, angle delta
        y = np.arctan2(-R[2, 0], sy)

        z = np.arctan2(R[1, 0], R[0, 0])  # atan2(R21,R11) -> ROLL, angle phi

    else:  # if in gymbal lock, use different formula for yaw, pitch roll
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])  # returns YAW, PITCH, ROLL angles in radians


def draw_pose_info(frame, img_point, point_proj, roll=None, pitch=None, yaw=None):
    frame = cv2.line(frame, img_point, tuple(
        point_proj[0].ravel().astype(int)), (255, 0, 0), 3)
    frame = cv2.line(frame, img_point, tuple(
        point_proj[1].ravel().astype(int)), (0, 255, 0), 3)
    frame = cv2.line(frame, img_point, tuple(
        point_proj[2].ravel().astype(int)), (0, 0, 255), 3)

    if roll is not None and pitch is not None and yaw is not None:
        cv2.putText(frame, "Roll:" + str(round(roll, 0)), (500, 50),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Pitch:" + str(round(pitch, 0)), (500, 70),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Yaw:" + str(round(yaw, 0)), (500, 90),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
