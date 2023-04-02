import cv2
import numpy as np
import cvxpy as cp


def get_smoothing_constraints(F, p, e, t):

    B_t = cp.bmat([[p[t - 2, 2], p[t - 2, 3], p[t - 2, 0]], [p[t - 2, 4], p[t - 2, 5], p[t - 2, 1]], [0, 0, 1]])
    B_t1 = cp.bmat(
        [[p[t - 1, 2], p[t - 1, 3], p[t - 1, 0]], [p[t - 1, 4], p[t - 1, 5], p[t - 1, 1]], [0, 0, 1]])
    B_t2 = cp.bmat(
        [[p[t, 2], p[t, 3], p[t, 0]], [p[t, 4], p[t, 5], p[t, 1]], [0, 0, 1]])
    B_t3 = cp.bmat(
        [[p[t + 1, 2], p[t + 1, 3], p[t + 1, 0]], [p[t + 1, 4], p[t + 1, 5], p[t + 1, 1]], [0, 0, 1]])

    R_t2 = cp.vec((F[t + 1] @ B_t3 - B_t2)[0:2])
    R_t1 = cp.vec((F[t] @ B_t2 - B_t1)[0:2])
    R_t = cp.vec((F[t - 1] @ B_t1 - B_t)[0:2])

    smooth_constraints = [R_t >= -e[0][t], R_t <= e[0][t], e[0][t] >= 0,
                          R_t1 - R_t >= -e[1][t], R_t1 - R_t <= e[1][t], e[1][t] >= 0,
                          R_t2 - 2 * R_t1 + R_t >= -e[2][t], R_t2 - 2 * R_t1 + R_t <= e[2][t], e[2][t] >= 0]

    return smooth_constraints


def get_proximity_constraints(p, t):

    proximity_constraints = [p[t, 2] >= 0.9, p[t, 5] >= 0.9,
                             p[t, 2] <= 1.1, p[t, 5] <= 1.1,
                             p[t, 3] >= -0.1, p[t, 4] >= -0.1,
                             p[t, 3] <= 0.1, p[t, 4] <= 0.1,
                             p[t, 3] + p[t, 4] >= -0.05, p[t, 3] + p[t, 4] <= 0.05,
                             p[t, 2] - p[t, 5] >= -0.1, p[t, 2] - p[t, 5] <= 0.1]

    return proximity_constraints


def get_inclusion_constraints(p, corners, shape, t):

    corner_1 = cp.vec(cp.bmat([[1, 0, corners[0, 0, 0], corners[0, 0, 1], 0, 0],
                        [0, 1, 0, 0, corners[0, 0, 0], corners[0, 0, 1]]]) @ p[t])
    corner_2 = cp.vec(cp.bmat([[1, 0, corners[1, 0, 0], corners[1, 0, 1], 0, 0],
                        [0, 1, 0, 0, corners[1, 0, 0], corners[1, 0, 1]]]) @ p[t])
    corner_3 = cp.vec(cp.bmat([[1, 0, corners[2, 0, 0], corners[2, 0, 1], 0, 0],
                        [0, 1, 0, 0, corners[2, 0, 0], corners[2, 0, 1]]]) @ p[t])
    corner_4 = cp.vec(cp.bmat([[1, 0, corners[3, 0, 0], corners[3, 0, 1], 0, 0],
                        [0, 1, 0, 0, corners[3, 0, 0], corners[3, 0, 1]]]) @ p[t])

    upper_bound = cp.vec([shape[1], shape[0]])
    lower_bound = cp.vec([0, 0])

    inclusion_constraints = [corner_1 <= upper_bound, corner_1 >= lower_bound,
                             corner_2 <= upper_bound, corner_2 >= lower_bound,
                             corner_3 <= upper_bound, corner_3 >= lower_bound,
                             corner_4 <= upper_bound, corner_4 >= lower_bound]

    return inclusion_constraints


def get_constraints(F, p, e, corners, shape, t):

    smoothing_constraints = get_smoothing_constraints(F, p, e, t)
    proximity_constraints = get_proximity_constraints(p, t)
    inclusion_constraints = get_inclusion_constraints(p, corners, shape, t)

    return smoothing_constraints, proximity_constraints, inclusion_constraints


def p_to_B(p, window):

    B = np.zeros((window, 3, 3))
    if len(p.shape) == 1:
        p = p[np.newaxis, :]

    B[:, 0, 0] = p[:, 2]
    B[:, 0, 1] = p[:, 3]
    B[:, 0, 2] = p[:, 0]
    B[:, 1, 0] = p[:, 4]
    B[:, 1, 1] = p[:, 5]
    B[:, 1, 2] = p[:, 1]
    B[:, 2, 2] = 1

    return B.squeeze()

def smooth_path(I, batch=30, crop_ratio=.7, qualityLevel=0.05):
    """
    :param I: list<np.ndarray> of video stills (shape=(num_frames x height x width))
    :return: C: np.ndarray containing original camera path (shape=(num_frames x 3 x 3))
    :return: P: np.ndarray containing smoothed camera path (shape=(num_frames x 3 x 3))
    :return: B: np.ndarray containing smoothing homography at each time step (shape=(num_frames x 3 x 3))

    Iterate through each pair of frames, denote I[t] and I[t+1]:
        Find F[t+1]: the relative motion homography from I[t+1] to I[t]
        Find C[t+1]: the absolute motion of camera at t+1 (i.e. F[t+1] @ C[t])

        *** LP ***
        B[t+1] represents an unknown smoothing matrix to solve for (shape=(3 x 3))
        Set smoothing constraints w/r/t path:
            R[t] = F[t+1] @ B[t+1] - B[t]
            Set unknown bounding variables e for each order derivative  (shape=(6x1))
        Set proximity constraints:
            0.9 <= a_t, d_t <= 1.1
            -0.1 <= b_t, c_t <= 0.1
            -0.05 <= b_c + c_t <= 0.05
            -0.01 <= a_t - d_t <= 0.1
        Set inclusion constraints:
            Compute new corners of frame at each t
            Make sure corners are within 0 and original height, width

        Solve LP
        Take solved B[t+1] and find P[t+1] = C[t+1] @ B[t+1]
    """

    # F: relative motion
    F = np.zeros((len(I), 3, 3))
    F[0] = np.identity(3)

    # C: absolute motion
    C = np.copy(F)

    # B: smoothing matrices
    B = np.copy(F)
    B[:] = np.identity(3)

    # P: smooth motion
    P = np.copy(B)

    # c: parameter weights
    c = np.asarray([1, 1, 100, 100, 100, 100])

    # w: objective weights
    w = np.asarray([10, 1, 100])

    corners = np.zeros((4, 1, 2), dtype=np.float64)
    y, x, _ = I[0].shape
    crop = ((1-crop_ratio) * np.asarray(I[0].shape[0:2])).astype(dtype=int)
    corners[0, 0] = [crop[1]//2, crop[0]//2]
    corners[1, 0] = [crop[1]//2, y - crop[0]//2]
    corners[2, 0] = [x - crop[1]//2, crop[0]//2]
    corners[3, 0] = [x - crop[1]//2, y - crop[0]//2]

    smoothing_constraints = []
    proximity_constraints = []
    inclusion_constraints = []
    objective = 0

    window = min(batch, len(I) - 3)

    p = cp.Variable((len(I), 6))
    e1 = cp.Variable((len(I), 6))
    e2 = cp.Variable((len(I), 6))
    e3 = cp.Variable((len(I), 6))
    e = [e1, e2, e3]

    # Parameters for feature matching (tuned from baseline values in references)
    ft_params = dict(maxCorners=100,
                     qualityLevel=qualityLevel,
                     minDistance=2,
                     blockSize=15)

    # Parameters for optical flow (tuned from baseline values in references)
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    i = -2

    for t in range(0, len(I) - 1):

        # Convert to true grayscale
        image_1 = np.round(np.mean(I[t].astype(dtype=np.float64), axis=-1)).astype(dtype=np.uint8)
        image_2 = np.round(np.mean(I[t + 1].astype(dtype=np.float64), axis=-1)).astype(dtype=np.uint8)

        points_1 = cv2.goodFeaturesToTrack(image_1, mask=None, **ft_params)
        points_2, status, _ = cv2.calcOpticalFlowPyrLK(image_2, image_1, points_1, None, **lk_params)

        successful = False

        if points_1 is not None and points_2 is not None:

            # Filter to only good points
            points_1 = points_1[status == 1].astype(np.uint8)
            points_2 = points_2[status == 1].astype(np.uint8)

            # If too many points were bad, consider frame an outlier
            if points_1.shape[0] > .5 * ft_params['maxCorners']:

                F[t + 1] = np.concatenate((cv2.estimateAffinePartial2D(points_1, points_2,
                                                                       cv2.RANSAC, ransacReprojThreshold=5.0)[0],
                                           np.asarray([[0, 0, 1]])), axis=0)

                successful = True

        if not successful:
            F[t + 1] = np.identity(3)

        C[t + 1] = F[t + 1] @ C[t]

        # Smoothing requires 3 frames of data
        if t >= 2:

            sc, pc, ic = get_constraints(F, p, e, corners, I[0].shape, t)
            smoothing_constraints += sc
            proximity_constraints += pc
            inclusion_constraints += ic
            objective += w[0] * (c @ e1[t]) + w[1] * (c @ e2[t]) + w[2] * (c @ e3[t])

        if i > 0 and i % window == window - 1:

            # proximity_constraints += get_proximity_constraints(p, i + 1) + get_proximity_constraints(p, i + 2) \
            #                          + get_proximity_constraints(p, i + 3)
            # inclusion_constraints += get_inclusion_constraints(p, corners, I[0].shape, i + 1) \
            #                          + get_inclusion_constraints(p, corners, I[0].shape, i + 2) \
            #                          + get_inclusion_constraints(p, corners, I[0].shape, i + 3)

            constraints = smoothing_constraints + proximity_constraints + inclusion_constraints

            obj = cp.Minimize(cp.sum(objective))
            prob = cp.Problem(obj, constraints)

            prob.solve(verbose=False)

            if prob.status not in ['infeasible', 'unbounded']:
                pt = np.asarray(p.value)
                B[t + 2 - window: t + 2] = p_to_B(pt[t + 2 - window: t + 2, :], window)

            smoothing_constraints = []
            proximity_constraints = []
            inclusion_constraints = []
            objective = 0

            for j in range(t - window, t + 1):
                smoothing_constraints += get_smoothing_constraints(F, p, e, j)
                proximity_constraints += get_proximity_constraints(p, j)
                inclusion_constraints += get_inclusion_constraints(p, corners, I[0].shape, j)
                objective += w[0] * (c @ e1[j]) + w[1] * (c @ e2[j]) + w[2] * (c @ e3[j])

            if t < len(I) - 1:
                window = min(batch, len(I) - t - 2)

            i = -1
        i += 1

    P = C @ B

    return C, P, B


def stabilize(frames, B, crop_ratio=.7):

    reds = []
    smooth_frames = []
    corners = np.zeros((4, 1, 2), dtype=np.float64)
    y, x, _ = frames[0].shape
    crop = ((1 - crop_ratio) * np.asarray(frames[0].shape[0:2])).astype(dtype=int)
    keep = (crop_ratio * np.asarray(frames[0].shape[0:2])).astype(dtype=int)

    corners[0, 0] = [crop[1]//2, crop[0]//2]
    corners[1, 0] = [crop[1]//2, y - crop[0]//2]
    corners[2, 0] = [x - crop[1]//2, crop[0]//2]
    corners[3, 0] = [x - crop[1]//2, y - crop[0]//2]

    # max_x = 0
    # min_x = frames[0].shape[0]
    # max_y = 0
    # min_y = frames[0].shape[1]

    for i in range(2, len(frames) - 1):
        c = np.round(cv2.perspectiveTransform(corners, np.linalg.inv(B[i])).squeeze()).astype(dtype=int)
        red = np.copy(frames[i])
        cv2.line(red, tuple(c[0]), tuple(c[1]), (0, 0, 255), 5)
        cv2.line(red, tuple(c[0]), tuple(c[2]), (0, 0, 255), 5)
        cv2.line(red, tuple(c[1]), tuple(c[3]), (0, 0, 255), 5)
        cv2.line(red, tuple(c[2]), tuple(c[3]), (0, 0, 255), 5)
        reds.append(red)

        # frame = frames[i][max(0, center[1] - keep[0]//2):center[1] + keep[0]//2, max(0, center[0] - keep[1]//2):center[0] + keep[1]//2]
        # max_x = np.max([max_x, keep[0]//2 - frame.shape[0]//2])
        # min_x = np.min([min_x, keep[0]//2 + int(np.ceil(frame.shape[0]/2))])
        # max_y = np.max([max_y, keep[1]//2 - frame.shape[1]//2])
        # min_y = np.min([min_y, keep[1]//2 + int(np.ceil(frame.shape[1]/2))])
        # smoothed[keep[0]//2 - frame.shape[0]//2:keep[0]//2 + int(np.ceil(frame.shape[0]/2)), keep[1]//2 - frame.shape[1]//2:keep[1]//2 + int(np.ceil(frame.shape[1]/2))] = frame
        # smooth_frames.append(smoothed.astype(dtype=np.uint8))
        smooth_frames.append(cv2.warpPerspective(frames[i], np.linalg.inv(B[i]), (keep[1], keep[0])))

    # # final crop
    # for i in range(len(smooth_frames)):
    #     smooth_frames[i] = np.copy(smooth_frames[i][max_x:min_x, max_y:min_y])

    return reds, smooth_frames
