import math

class Reward:
    def __init__(self, verbose=False):
        self.first_racingpoint_index = None
        self.verbose = verbose

    def reward_function(self, params):

        # Import package (needed for heading)
        ################## HELPER FUNCTIONS ###################

        def dist_2_points(x1, x2, y1, y2):
            return abs(abs(x1-x2)**2 + abs(y1-y2)**2)**0.5
        # 最近的两个点的index，难道就不怕得到的是已经行驶过的两个点吗
        def closest_2_racing_points_index(racing_coords, car_coords):

            # Calculate all distances to racing points
            distances = []
            for i in range(len(racing_coords)):
                distance = dist_2_points(x1=racing_coords[i][0], x2=car_coords[0],
                                         y1=racing_coords[i][1], y2=car_coords[1])
                distances.append(distance)

            # Get index of the closest racing point
            closest_index = distances.index(min(distances))

            # Get index of the second closest racing point
            distances_no_closest = distances.copy()
            distances_no_closest[closest_index] = 999
            second_closest_index = distances_no_closest.index(
                min(distances_no_closest))

            return [closest_index, second_closest_index]

        # optimals[0:2]             最接近的一个点的xy坐标，speed
        # optimals_second[0:2]      倒数第二接近的一个点的xy坐标，speed
        # [x, y]                    当前坐标
        def dist_to_racing_line(closest_coords, second_closest_coords, car_coords):
            
            # 两个最近点之间的距离
            a = abs(dist_2_points(x1=closest_coords[0],
                                  x2=second_closest_coords[0],
                                  y1=closest_coords[1],
                                  y2=second_closest_coords[1]))

            # 当前小车，和最近点的距离
            b = abs(dist_2_points(x1=car_coords[0],
                                  x2=closest_coords[0],
                                  y1=car_coords[1],
                                  y2=closest_coords[1]))
            # 当前小车，和第二近点的距离
            c = abs(dist_2_points(x1=car_coords[0],
                                  x2=second_closest_coords[0],
                                  y1=car_coords[1],
                                  y2=second_closest_coords[1]))

            # Calculate distance between car and racing line (goes through 2 closest racing points)
            # try-except in case a=0 (rare bug in DeepRacer)
            try:
                distance = abs(-(a**4) + 2*(a**2)*(b**2) + 2*(a**2)*(c**2) -
                               (b**4) + 2*(b**2)*(c**2) - (c**4))**0.5 / (2*a)
            except:
                distance = b

            return distance

        # Calculate which one of the closest racing points is the next one and which one the previous one
        def next_prev_racing_point(closest_coords, second_closest_coords, car_coords, heading):

            # Virtually set the car more into the heading direction
            heading_vector = [math.cos(math.radians(
                heading)), math.sin(math.radians(heading))]
            new_car_coords = [car_coords[0]+heading_vector[0],
                              car_coords[1]+heading_vector[1]]

            # Calculate distance from new car coords to 2 closest racing points
            distance_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                        x2=closest_coords[0],
                                                        y1=new_car_coords[1],
                                                        y2=closest_coords[1])
            distance_second_closest_coords_new = dist_2_points(x1=new_car_coords[0],
                                                               x2=second_closest_coords[0],
                                                               y1=new_car_coords[1],
                                                               y2=second_closest_coords[1])

            if distance_closest_coords_new <= distance_second_closest_coords_new:
                next_point_coords = closest_coords
                prev_point_coords = second_closest_coords
            else:
                next_point_coords = second_closest_coords
                prev_point_coords = closest_coords

            return [next_point_coords, prev_point_coords]

        def racing_direction_diff(closest_coords, second_closest_coords, car_coords, heading):

            # Calculate the direction of the center line based on the closest waypoints
            next_point, prev_point = next_prev_racing_point(closest_coords,
                                                            second_closest_coords,
                                                            car_coords,
                                                            heading)

            # Calculate the direction in radius, arctan2(dy, dx), the result is (-pi, pi) in radians
            track_direction = math.atan2(
                next_point[1] - prev_point[1], next_point[0] - prev_point[0])

            # Convert to degree
            track_direction = math.degrees(track_direction)

            # Calculate the difference between the track direction and the heading direction of the car
            direction_diff = abs(track_direction - heading)
            if direction_diff > 180:
                direction_diff = 360 - direction_diff

            return direction_diff

        # Gives back indexes that lie between start and end index of a cyclical list 
        # (start index is included, end index is not)
        def indexes_cyclical(start, end, array_len):

            if end < start:
                end += array_len

            return [index % array_len for index in range(start, end)]

        # Calculate how long car would take for entire lap, if it continued like it did until now
        # first_index -> self.first_racingpoint_index 训练时，step=1，离当前最近的一个点的index
        # 然后在整个过程就不变，这个index会很小。eval时，一直是0
        # closest_index 离当前最近的一个点的index
        # step_count 我也不知道这个参数具体的意义，一共有多少个步骤数？
        # times_list 每一个点期待的通过时间
        def projected_time(first_index, closest_index, step_count, times_list):

            # Calculate how much time has passed since start
            current_actual_time = (step_count-1) / 15

            # Calculate which indexes were already passed
            indexes_traveled = indexes_cyclical(first_index, closest_index, len(times_list))

            # Calculate how much time should have passed if car would have followed optimals
            current_expected_time = sum([times_list[i] for i in indexes_traveled])

            # Calculate how long one entire lap takes if car follows optimals
            total_expected_time = sum(times_list)

            # Calculate how long car would take for entire lap, if it continued like it did until now
            try:
                projected_time = (current_actual_time/current_expected_time) * total_expected_time
            except:
                projected_time = 9999

            return projected_time

        #################### RACING LINE ######################

        # Optimal racing line for the Spain track
        # Each row: [x,y,speed,timeFromPreviousPoint]  eg:[0.34775, -2.173, 4.0, 0.07904]
        racing_track = [[5.56857, -0.24708, 4.0, 0.0668],
                        [5.57257, 0.00899, 4.0, 0.06402],
                        [5.57025, 0.25314, 4.0, 0.06104],
                        [5.56208, 0.4852, 4.0, 0.05805],
                        [5.54842, 0.70571, 4.0, 0.05523],
                        [5.52954, 0.91561, 4.0, 0.05269],
                        [5.50546, 1.11559, 4.0, 0.05036],
                        [5.47543, 1.3054, 3.8004, 0.05057],
                        [5.43821, 1.48499, 3.8004, 0.04826],
                        [5.392, 1.65517, 3.8004, 0.0464],
                        [5.33344, 1.81758, 3.8004, 0.04543],
                        [5.25499, 1.97645, 4.0, 0.0443],
                        [5.15692, 2.13642, 4.0, 0.04691],
                        [5.04182, 2.29656, 4.0, 0.0493],
                        [4.91484, 2.45437, 4.0, 0.05064],
                        [4.77906, 2.60821, 4.0, 0.0513],
                        [4.63675, 2.75733, 4.0, 0.05153],
                        [4.48942, 2.90159, 4.0, 0.05155],
                        [4.33804, 3.04111, 4.0, 0.05147],
                        [4.18332, 3.17621, 4.0, 0.05135],
                        [4.02567, 3.30708, 4.0, 0.05122],
                        [3.86528, 3.43381, 4.0, 0.05111],
                        [3.70212, 3.55626, 4.0, 0.051],
                        [3.5357, 3.67366, 4.0, 0.05091],
                        [3.36542, 3.78524, 4.0, 0.0509],
                        [3.18955, 3.88916, 4.0, 0.05107],
                        [3.00364, 3.98222, 4.0, 0.05197],
                        [2.79927, 4.06337, 4.0, 0.05497],
                        [2.57638, 4.1377, 4.0, 0.05874],
                        [2.33297, 4.20691, 4.0, 0.06326],
                        [2.06528, 4.27195, 4.0, 0.06887],
                        [1.76962, 4.33377, 4.0, 0.07551],
                        [1.44251, 4.39327, 4.0, 0.08312],
                        [1.08063, 4.45132, 4.0, 0.09163],
                        [0.68346, 4.50866, 4.0, 0.10032],
                        [0.24565, 4.56589, 4.0, 0.11039],
                        [-0.23623, 4.6236, 4.0, 0.12133],
                        [-0.76722, 4.68241, 4.0, 0.13356],
                        [-1.35596, 4.743, 4.0, 0.14796],
                        [-2.02812, 4.80683, 4.0, 0.1688],
                        [-2.90892, 4.88085, 4.0, 0.22098],
                        [-3.47287, 4.91019, 4.0, 0.14118],
                        [-3.86705, 4.91312, 4.0, 0.09855],
                        [-4.16789, 4.9001, 4.0, 0.07528],
                        [-4.4109, 4.87634, 4.0, 0.06104],
                        [-4.61435, 4.84468, 3.6378, 0.0566],
                        [-4.78753, 4.80656, 2.91667, 0.0608],
                        [-4.93558, 4.76271, 2.6313, 0.05868],
                        [-5.06213, 4.71364, 2.04763, 0.06629],
                        [-5.17055, 4.6599, 1.9, 0.06369],
                        [-5.25962, 4.60093, 1.9, 0.05623],
                        [-5.33198, 4.53728, 1.9, 0.05072],
                        [-5.38413, 4.46813, 1.9, 0.04559],
                        [-5.4178, 4.39471, 1.9013, 0.04248],
                        [-5.43524, 4.31862, 2.15709, 0.03619],
                        [-5.44063, 4.24107, 2.479, 0.03136],
                        [-5.43676, 4.16234, 2.77175, 0.02844],
                        [-5.42499, 4.08225, 3.14409, 0.02575],
                        [-5.40641, 4.00047, 3.44013, 0.02438],
                        [-5.38127, 3.91648, 3.81935, 0.02296],
                        [-5.34975, 3.82967, 4.0, 0.02309],
                        [-5.31217, 3.73956, 4.0, 0.02441],
                        [-5.2691, 3.64592, 4.0, 0.02577],
                        [-5.22155, 3.54906, 4.0, 0.02697],
                        [-5.17008, 3.44913, 4.0, 0.0281],
                        [-5.06131, 3.22482, 4.0, 0.06232],
                        [-4.94817, 2.97592, 4.0, 0.06835],
                        [-4.83061, 2.70183, 4.0, 0.07456],
                        [-4.70825, 2.40021, 4.0, 0.08137],
                        [-4.5806, 2.06824, 4.0, 0.08892],
                        [-4.44723, 1.70337, 4.0, 0.09712],
                        [-4.30778, 1.30381, 4.0, 0.1058],
                        [-4.16201, 0.86835, 4.0, 0.1148],
                        [-4.01014, 0.39843, 4.0, 0.12346],
                        [-3.85286, -0.10191, 4.0, 0.13112],
                        [-3.68989, -0.63403, 4.0, 0.13913],
                        [-3.52064, -1.20096, 4.0, 0.14791],
                        [-3.34228, -1.81762, 4.0, 0.16048],
                        [-3.12148, -2.54117, 4.0, 0.18912],
                        [-2.93814, -3.07195, 4.0, 0.14039],
                        [-2.78306, -3.46374, 4.0, 0.10534],
                        [-2.64819, -3.75545, 3.67036, 0.08756],
                        [-2.5263, -3.97908, 3.12603, 0.08147],
                        [-2.41255, -4.1524, 2.6045, 0.0796],
                        [-2.30246, -4.28966, 2.6045, 0.06756],
                        [-2.19227, -4.39876, 2.6045, 0.05954],
                        [-2.07774, -4.48379, 2.6045, 0.05477],
                        [-1.94999, -4.54449, 3.25544, 0.04345],
                        [-1.80786, -4.59093, 3.66893, 0.04075],
                        [-1.64838, -4.62455, 4.0, 0.04075],
                        [-1.46761, -4.64556, 4.0, 0.04549],
                        [-1.26016, -4.65321, 4.0, 0.0519],
                        [-1.0212, -4.64682, 4.0, 0.05976],
                        [-0.74663, -4.62549, 4.0, 0.06885],
                        [-0.43368, -4.58822, 4.0, 0.07879],
                        [-0.08181, -4.53413, 4.0, 0.089],
                        [0.30648, -4.46279, 4.0, 0.0987],
                        [0.72524, -4.37447, 4.0, 0.10699],
                        [1.1653, -4.27036, 4.0, 0.11305],
                        [1.61547, -4.15231, 4.0, 0.11635],
                        [2.06408, -4.02317, 4.0, 0.11671],
                        [2.5017, -3.88568, 4.0, 0.11468],
                        [2.91966, -3.74185, 4.0, 0.11051],
                        [3.30883, -3.59357, 4.0, 0.10412],
                        [3.66066, -3.44278, 4.0, 0.0957],
                        [3.97158, -3.29116, 4.0, 0.08648],
                        [4.24287, -3.13949, 4.0, 0.0777],
                        [4.47028, -2.98752, 4.0, 0.06838],
                        [4.67226, -2.83463, 3.46867, 0.07303],
                        [4.84827, -2.67968, 3.46867, 0.0676],
                        [4.99884, -2.52079, 3.46867, 0.06311],
                        [5.12427, -2.35478, 3.46867, 0.05999],
                        [5.22062, -2.17277, 4.0, 0.05148],
                        [5.30163, -1.97741, 4.0, 0.05287],
                        [5.36971, -1.76821, 4.0, 0.055],
                        [5.42621, -1.54446, 4.0, 0.05769],
                        [5.4725, -1.30613, 4.0, 0.0607],
                        [5.50945, -1.05307, 4.0, 0.06394],
                        [5.53765, -0.78647, 4.0, 0.06702],
                        [5.5573, -0.51404, 4.0, 0.06828]]

        ################## INPUT PARAMETERS ###################

        # Read all input parameters
        all_wheels_on_track = params['all_wheels_on_track']
        x = params['x']
        y = params['y']
        distance_from_center = params['distance_from_center']
        is_left_of_center = params['is_left_of_center']
        heading = params['heading']
        progress = params['progress']
        steps = params['steps']
        speed = params['speed']
        steering_angle = params['steering_angle']
        track_width = params['track_width']
        waypoints = params['waypoints']
        closest_waypoints = params['closest_waypoints']
        is_offtrack = params['is_offtrack']

        ############### OPTIMAL X,Y,SPEED,TIME ################

        # Get closest indexes for racing line (and distances to all points on racing line)
        closest_index, second_closest_index = closest_2_racing_points_index(
            racing_track, [x, y])

        # Get optimal [x, y, speed, time] for closest and second closest index
        optimals = racing_track[closest_index]
        optimals_second = racing_track[second_closest_index]

        # Save first racingpoint of episode for later
        if self.verbose == True:
            self.first_racingpoint_index = 0 # this is just for testing purposes
        if steps == 1:
            self.first_racingpoint_index = closest_index

        ################ REWARD AND PUNISHMENT ################

        ## Define the default reward ##
        reward = 1

        ## Reward if car goes close to optimal racing line ##
        DISTANCE_MULTIPLE = 1
        dist = dist_to_racing_line(optimals[0:2], optimals_second[0:2], [x, y])
        distance_reward = max(1e-3, 1 - (dist/(track_width*0.5)))#dist是小车和赛道的距离，越近dist越小，dist/(track_width*0.5)就越接近0
        reward += distance_reward * DISTANCE_MULTIPLE

        ## Reward if speed is close to optimal speed ##
        SPEED_DIFF_NO_REWARD = 1
        SPEED_MULTIPLE = 2
        speed_diff = abs(optimals[2]-speed)#下一个点应有的速度-当前速度
        if speed_diff <= SPEED_DIFF_NO_REWARD:
            # we use quadratic punishment (not linear) bc we're not as confident with the optimal speed
            # so, we do not punish small deviations from optimal speed
            # 我们使用二次惩罚（非线性），因为我们对最佳速度没有信心。 所以，我们不会惩罚与最佳速度的小偏差
            speed_reward = (1 - (speed_diff/(SPEED_DIFF_NO_REWARD))**2)**2
        else:
            speed_reward = 0
        reward += speed_reward * SPEED_MULTIPLE

        # Reward if less steps
        REWARD_PER_STEP_FOR_FASTEST_TIME = 1 
        STANDARD_TIME = 37 #最好成绩是47s，而不是37
        FASTEST_TIME = 27
        times_list = [row[3] for row in racing_track]
        #projected_time是根据过去表现，算出来会花费的通过时间
        projected_time = projected_time(self.first_racingpoint_index, closest_index, steps, times_list)
        try:
            steps_prediction = projected_time * 15 + 1#预计总共的步骤数
            reward_prediction = max(1e-3, (-REWARD_PER_STEP_FOR_FASTEST_TIME*(FASTEST_TIME) /
                                           (STANDARD_TIME-FASTEST_TIME))*(steps_prediction-(STANDARD_TIME*15+1)))
            steps_reward = min(REWARD_PER_STEP_FOR_FASTEST_TIME, reward_prediction / steps_prediction)
        except:
            steps_reward = 0
        reward += steps_reward

        # Zero reward if obviously wrong direction (e.g. spin)
        direction_diff = racing_direction_diff(
            optimals[0:2], optimals_second[0:2], [x, y], heading)
        if direction_diff > 30:
            reward = 1e-3
            
        # Zero reward of obviously too slow
        speed_diff_zero = optimals[2]-speed
        if speed_diff_zero > 0.5:
            reward = 1e-3
            
        ## Incentive for finishing the lap in less steps ##
        REWARD_FOR_FASTEST_TIME = 1500 # should be adapted to track length and other rewards
        STANDARD_TIME = 37  # seconds (time that is easily done by model)
        FASTEST_TIME = 27  # seconds (best time of 1st place on the track)
        if progress == 100:
            finish_reward = max(1e-3, (-REWARD_FOR_FASTEST_TIME /
                      (15*(STANDARD_TIME-FASTEST_TIME)))*(steps-STANDARD_TIME*15))
        else:
            finish_reward = 0
        reward += finish_reward
        
        ## Zero reward if off track ##
        if all_wheels_on_track == False:
            reward = 1e-3

        ####################### VERBOSE #######################
        
        if self.verbose == True:
            print("Closest index: %i" % closest_index)
            print("Distance to racing line: %f" % dist)
            print("=== Distance reward (w/out multiple): %f ===" % (distance_reward))
            print("Optimal speed: %f" % optimals[2])
            print("Speed difference: %f" % speed_diff)
            print("=== Speed reward (w/out multiple): %f ===" % speed_reward)
            print("Direction difference: %f" % direction_diff)
            print("Predicted time: %f" % projected_time)
            print("=== Steps reward: %f ===" % steps_reward)
            print("=== Finish reward: %f ===" % finish_reward)
            
        #################### RETURN REWARD ####################
        
        # Always return a float value
        return float(reward)


reward_object = Reward() # add parameter verbose=True to get noisy output for testing


def reward_function(params):
    return reward_object.reward_function(params)
