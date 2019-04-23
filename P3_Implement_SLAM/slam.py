import numpy as np 

def initialize_constraints(N, num_landmarks, world_size):
    ''' This function takes in a number of time steps N, number of landmarks, and a world_size,
        and returns initialized constraint matrices, omega and xi.'''
    
    ## Recommended: Define and store the size (rows/cols) of the constraint matrix in a variable
    
    ## TODO: Define the constraint matrix, Omega, with two initial "strength" values
    ## for the initial x, y location of our robot
    size = (N+num_landmarks) * 2
    omega = np.zeros((size, size))
    
    ## TODO: Define the constraint *vector*, xi
    ## you can assume that the robot starts out in the middle of the world with 100% confidence
    xi = np.zeros((size, 1))
    
    # Initialize in the Middle of the world with 100% confidence
    worldMid = world_size / 2.0
    omega[0][0] = 1
    omega[1][1] = 1
    xi[0][0] = worldMid
    xi[1][0] = worldMid
    
    return omega, xi

## TODO: Complete the code to implement SLAM
def OmegaUpdate(omega, i, i2, strength=1):
    omega[i][i] += strength
    omega[i][i2] += -strength
    omega[i2][i] += -strength
    omega[i2][i2] += strength

def UpdateMeasurements(omega, xi, measurements, N, noise):
    noise = 1.0 / noise
    # measurements[[locIdx, dx, dy]...]
    print('measurements:', len(measurements))
    for s, measures in enumerate(measurements):
        t = 2 * s
        for locIdx, dx, dy in measures:
            OmegaUpdate(omega, t, 2*(locIdx+N), noise)              # Omega Update X
            OmegaUpdate(omega, t+1, 2*(locIdx+N)+1, noise)      # Omega Update Y
            # Xi Update dX
            xi[t] += -dx * noise
            xi[2*(locIdx+N)] += dx * noise
            # Xi Update dY
            xi[t+1] += -dy * noise
            xi[2*(locIdx+N)+1] += dy * noise
    
def UpdateMovements(omega, xi, movements, noise):
    noise = 1.0 / noise
    for p, (dx, dy) in enumerate(movements):
        # Update Omega & Xi on X axis
        idx = np.array([2*p, 2*p+2])
        OmegaUpdate(omega, idx[0], idx[1], noise)
        xi[idx] += np.array([-dx, dx]) * noise
        # Update Omega & Xi on Y axis
        idx += 1
        OmegaUpdate(omega, idx[0], idx[1], noise)
        xi[idx] += np.array([-dy, dy]) * noise
        #omega[i:i2, i:i2] += strength
        #xi[i:i2] += [-dx, dx]
        ## Update Omega & Xi on Y axis
        #i, i2 = i+dim, i2+dim
        #omega[i:i2, i:i2] += strength
        #xi[i:i2] += [-dy, dy]
        
## slam takes in 6 arguments and returns mu, 
## mu is the entire path traversed by a robot (all x,y poses) *and* all landmarks locations
def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise):
    ## TODO: Use your initilization to create constraint matrices, omega and xi
    omega, xi = initialize_constraints(N, num_landmarks, world_size)
    dim = N + num_landmarks
    xi = xi.squeeze()
    
    ## TODO: Iterate through each time step in the data
    ## get all the motion and measurement data as you iterate
    measurements, motions = [], []
    for measure, mov in data:
        measurements.append(measure)
        motions.append(mov)
    
    ## TODO: update the constraint matrix/vector to account for all *measurements*
    ## this should be a series of additions that take into account the measurement noise
    UpdateMeasurements(omega, xi, measurements, N, measurement_noise)
            
    ## TODO: update the constraint matrix/vector to account for all *motion* and motion noise
    UpdateMovements(omega, xi, motions, motion_noise)
    
    ## TODO: After iterating through all the data
    ## Compute the best estimate of poses and landmark positions
    ## using the formula, omega_inverse * Xi
    xi = xi.reshape((-1, 1))
    omega_inverse = np.linalg.inv(np.matrix(omega))
    mu = omega_inverse * xi
    
    return mu # return `mu`

# a helper function that creates a list of poses and of landmarks for ease of printing
# this only works for the suggested constraint architecture of interlaced x,y poses
def get_poses_landmarks(mu, N, num_landmarks):
    # create a list of poses
    poses = []
    for i in range(N):
        poses.append((mu[2*i].item(), mu[2*i+1].item()))

    # create a list of landmarks
    landmarks = []
    for i in range(num_landmarks):
        landmarks.append((mu[2*(N+i)].item(), mu[2*(N+i)+1].item()))

    # return completed lists
    return poses, landmarks

def print_all(poses, landmarks):
    print('\n')
    print('Estimated Poses:')
    for i in range(len(poses)):
        print('['+', '.join('%.3f'%p for p in poses[i])+']')
    print('\n')
    print('Estimated Landmarks:')
    for i in range(len(landmarks)):
        print('['+', '.join('%.3f'%l for l in landmarks[i])+']')

if __name__ == '__main__':
    # Here is the data and estimated outputs for test case 1
    test_data1 = [[[[1, 19.457599255548065, 23.8387362100849], [2, -13.195807561967236, 11.708840328458608], [3, -30.0954905279171, 15.387879242505843]], [-12.2607279422326, -15.801093326936487]], [[[2, -0.4659930049620491, 28.088559771215664], [4, -17.866382374890936, -16.384904503932]], [-12.2607279422326, -15.801093326936487]], [[[4, -6.202512900833806, -1.823403210274639]], [-12.2607279422326, -15.801093326936487]], [[[4, 7.412136480918645, 15.388585962142429]], [14.008259661173426, 14.274756084260822]], [[[4, -7.526138813444998, -0.4563942429717849]], [14.008259661173426, 14.274756084260822]], [[[2, -6.299793150150058, 29.047830407717623], [4, -21.93551130411791, -13.21956810989039]], [14.008259661173426, 14.274756084260822]], [[[1, 15.796300959032276, 30.65769689694247], [2, -18.64370821983482, 17.380022987031367]], [14.008259661173426, 14.274756084260822]], [[[1, 0.40311325410337906, 14.169429532679855], [2, -35.069349468466235, 2.4945558982439957]], [14.008259661173426, 14.274756084260822]], [[[1, -16.71340983241936, -2.777000269543834]], [-11.006096015782283, 16.699276945166858]], [[[1, -3.611096830835776, -17.954019226763958]], [-19.693482634035977, 3.488085684573048]], [[[1, 18.398273354362416, -22.705102332550947]], [-19.693482634035977, 3.488085684573048]], [[[2, 2.789312482883833, -39.73720193121324]], [12.849049222879723, -15.326510824972983]], [[[1, 21.26897046581808, -10.121029799040915], [2, -11.917698965880655, -23.17711662602097], [3, -31.81167947898398, -16.7985673023331]], [12.849049222879723, -15.326510824972983]], [[[1, 10.48157743234859, 5.692957082575485], [2, -22.31488473554935, -5.389184118551409], [3, -40.81803984305378, -2.4703329790238118]], [12.849049222879723, -15.326510824972983]], [[[0, 10.591050242096598, -39.2051798967113], [1, -3.5675572049297553, 22.849456408289125], [2, -38.39251065320351, 7.288990306029511]], [12.849049222879723, -15.326510824972983]], [[[0, -3.6225556479370766, -25.58006865235512]], [-7.8874682868419965, -18.379005523261092]], [[[0, 1.9784503557879374, -6.5025974151499]], [-7.8874682868419965, -18.379005523261092]], [[[0, 10.050665232782423, 11.026385307998742]], [-17.82919359778298, 9.062000642947142]], [[[0, 26.526838150174818, -0.22563393232425621], [4, -33.70303936886652, 2.880339841013677]], [-17.82919359778298, 9.062000642947142]]]

    ### Uncomment the following three lines for test case 1 and compare the output to the values above ###
    num_landmarks = 5
    mu_1 = slam(test_data1, 20, num_landmarks, 100.0, 2.0, 2.0)
    poses, landmarks = get_poses_landmarks(mu_1, 20, num_landmarks)
    print_all(poses, landmarks)

    ##  Test Case 1
    ##
    # Estimated Pose(s):
    #     [50.000, 50.000]
    #     [37.858, 33.921]
    #     [25.905, 18.268]
    #     [13.524, 2.224]
    #     [27.912, 16.886]
    #     [42.250, 30.994]
    #     [55.992, 44.886]
    #     [70.749, 59.867]
    #     [85.371, 75.230]
    #     [73.831, 92.354]
    #     [53.406, 96.465]
    #     [34.370, 100.134]
    #     [48.346, 83.952]
    #     [60.494, 68.338]
    #     [73.648, 53.082]
    #     [86.733, 38.197]
    #     [79.983, 20.324]
    #     [72.515, 2.837]
    #     [54.993, 13.221]
    #     [37.164, 22.283]

    # Estimated Landmarks:
    #     [82.679, 13.435]
    #     [70.417, 74.203]
    #     [36.688, 61.431]
    #     [18.705, 66.136]
    #     [20.437, 16.983]