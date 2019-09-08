import math
import numpy as np
import scipy as scipy
from numpy.random import uniform
import scipy.stats

np.set_printoptions(threshold=3) # 打印临界值为3，超过临界值就会缩写
np.set_printoptions(suppress=True) # 过小的数据会被压缩
import cv2


def drawLines(img, points, r, g, b):
    cv2.polylines(img, [np.int32(points)], isClosed=False, color=(r, g, b))


def drawCross(img, center, r, g, b):
    d = 5
    t = 2
    LINE_AA = cv2.CV_AA if cv2.__version__[0] == '3' else cv2.LINE_AA
    color = (r, g, b)
    ctrx = center[0, 0]
    ctry = center[0, 1]
    cv2.line(img, (ctrx - d, ctry - d), (ctrx + d, ctry + d), color, t, LINE_AA)
    cv2.line(img, (ctrx + d, ctry - d), (ctrx - d, ctry + d), color, t, LINE_AA)


def mouseCallback(event, x, y, flags, null):
    global center
    global trajectory
    global previous_x
    global previous_y
    global zs
    global landmarks
    global lii
    global li
    global ti
    global loca

    #改造1：周期性移动landmark 实现右移50次之后返回
    lii += 1
    if lii <= 50:
        li = 1
    else:
        li = -1
        ti += 1
        if ti == 51:
            lii = 0
            ti = 0
    landmarks[:, 0] += li
    center = np.array([[x, y]])
    trajectory = np.vstack((trajectory, np.array([x, y])))
    # noise=sensorSigma * np.random.randn(1,2) + sensorMu

    if previous_x > 0:
        heading = np.arctan2(np.array([y - previous_y]), np.array([previous_x - x]))

        if heading > 0:
            heading = -(heading - np.pi)
        else:
            heading = -(np.pi + heading)

        distance = np.linalg.norm(np.array([[previous_x, previous_y]]) - np.array([[x, y]]), axis=1)
        std = np.array([2, 4])
        u = np.array([heading, distance])
        zs = (np.linalg.norm(landmarks - center, axis=1) + (np.random.randn(NL) * sensor_std_err))
        predict(particles, u, std, dt=1.)
        update(particles, weights, z=zs, R=50, landmarks=landmarks)

        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)

        # 改造3：相对于直接求平均值，带权平均值更优
        loca = np.average(particles,weights=weights,axis=0)

    previous_x = x
    previous_y = y

loca = [0,0]
WIDTH = 800
HEIGHT = 600
WINDOW_NAME = "Particle Filter"

# sensorMu=0
# sensorSigma=3

sensor_std_err = 5

def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    return particles


def predict(particles, u, std, dt=1.):
    N = len(particles)
    # dist= (u[1] * dt) + (np.random.poisson(lam = 0.,size = N))
    # 本意是测量值的噪声来自于泊松分布，以便于在权值估计中有合理的理由使用泊松分布，但是设置最优偏差都不行...
    # 泊松分布得到的随机变量大概过于大了使得得到的粒子坐标超出屏幕之外
    # 或许这里算是一个为解决问题罢了，所以测量函数的噪声过程还是使用了正态分布
    dist = (u[1] * dt) + (np.random.randn(N) * std[1])
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist


def update(particles, weights, z, R, landmarks):
    weights.fill(1.)
    N = len(weights)
    disi = np.zeros(N, 'i')
    for i, landmark in enumerate(landmarks):
        distance = np.power((particles[:, 0] - landmark[0]) ** 2 + (particles[:, 1] - landmark[1]) ** 2, 0.5)
        #改造2：这一步原因：为了更随机的取得整型的泊松分布变量，使用一个随机数来选择向上或者向下取整使得结果较优，
                        # 类比数据结构快排的partition()函数的方法
        for j in range(0,N):
            jj = np.random.rand(0,1)
            if jj < 0.5:
                disi[j] = int(distance[j])
            else:
                disi[j] = math.ceil(distance[j])

        weights *= scipy.stats.poisson.pmf(disi, z[i]) #期望为Z[i]时，测量值为disi的概率
        #weights *= scipy.stats.norm(distance, R).pdf(z[i])
    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)


def neff(weights):
    return 1. / np.sum(np.square(weights))


def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N and j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def estimate(particles, weights):
    pos = particles[:, 0:1]
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)


x_range = np.array([0, 800])
y_range = np.array([0, 600])

# Number of partciles
N = 400
li = 0

landmarks = np.array([[144, 73], [410, 13], [336, 175], [718, 159], [178, 484], [665, 464]])
NL = len(landmarks)
particles = create_uniform_particles(x_range, y_range, N)

weights = np.array([1.0] * N)

# Create a black image, a window and bind the function to window
img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, mouseCallback)

center = np.array([[-10, -10]])

trajectory = np.zeros(shape=(0, 2))
robot_pos = np.zeros(shape=(0, 2))
previous_x = -1
previous_y = -1
DELAY_MSEC = 50
lii = 0
ti = 0

while (1):

    cv2.imshow(WINDOW_NAME, img)
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    drawLines(img, trajectory, 0, 255, 0)
    drawCross(img, center, r=255, g=0, b=0)

    # landmarks
    for landmark in landmarks:
        cv2.circle(img, tuple(landmark), 10, (255, 0, 0), -1)

    # draw_particles:
    #粒子初始位置，为了凸显平均位置，隐藏了移动过程白色粒子
    if loca[0] == 0 and loca[1] == 0:
        for particle in particles:
            cv2.circle(img, tuple((int(particle[0]), int(particle[1]))), 1, (255, 255, 255), -1)

    #the red dot is the average position of the particals
    cv2.circle(img, tuple((int(loca[0]), int(loca[1]))), 2, (0, 0, 255), -1)


    if cv2.waitKey(DELAY_MSEC) & 0xFF == 27:
        break

    cv2.circle(img, (10, 10), 10, (255, 0, 0), -1)
    cv2.circle(img, (10, 30), 3, (255, 255, 255), -1)
    cv2.putText(img, "Landmarks", (30, 20), 1, 1.0, (255, 0, 0))
    cv2.putText(img, "Particles", (30, 40), 1, 1.0, (255, 255, 255))
    cv2.putText(img, "Robot Trajectory(Ground truth)", (30, 60), 1, 1.0, (0, 255, 0))
    drawLines(img, np.array([[10, 55], [25, 55]]), 0, 255, 0)

cv2.destroyAllWindows()