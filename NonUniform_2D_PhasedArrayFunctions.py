# Imports

import pandas as pd
import random
import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy.fft import fft,fftfreq
# Import Library
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import os, sys
from matplotlib.lines import Line2D

"""
Section 1: 
----------
Setting up the classes for the individual antennas and the microwave sources/emitters
"""


class Antenna():
    # This holds the 3D location of the antenna. The antennas live in the x-y plane
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class Source():
    # This holds the frequency, the speed of light and the 3D position of the microwave source/emmiter
    def __init__(self, freq, speed, source_pos):
        self.freq = freq
        self.speed = speed
        self.source_pos = source_pos

    # This function will calculate the amplitude of the EM signal given the distance between the wave source and a particular location.
    # This was not actually used in the end
    def sig_at_time_and_location(self, pos, time):
        x1, y1, z1 = pos
        x0, y0, z0 = self.source_pos
        d = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2)
        pi2f = 2 * np.pi * self.freq
        wavelength = self.speed / self.freq
        sig = np.sin((2 * np.pi * d) / wavelength + 2 * np.pi * self.freq * time) / d
        return sig


"""
Section 2: 
----------
First let us build the individual rectangular models for the 3 components in question.
- The PCB - Printed circuit board
- The SMP connectors - Subminiature push-on connectors (can deal with < 40 GHz)
- The WWCs - WithWave connectors

These functions will generate 4 points corresponding to the corners of each rectangle
"""


def generate_PCB_rect(ant_pos, dc_orien):
    center_offset = 12.5e-1 * (1 if dc_orien == 1 else -1)
    cent_x, cent_y = ant_pos[0] + center_offset, ant_pos[1]
    hw, hh = PCB_width / 2, PCB_height / 2
    return [(cent_x + dx * hw, cent_y + dy * hh) for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]


def generate_SMP_rect(ant_pos, dc_orien):
    center_offset = 28e-1 * (1 if dc_orien == 1 else -1)
    cent_x, cent_y = ant_pos[0] + center_offset, ant_pos[1] + (5e-1 - PCB_height / 2) * (1 if dc_orien == 1 else -1)
    hw, hh = SMP_width / 2, SMP_height / 2
    return [(cent_x + dx * hw, cent_y + dy * hh) for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]


def generate_WWC_rect(ant_pos, dc_orien):
    center_offset_x = (1.3 - WWC_width / 2) * (1 if dc_orien == 1 else -1)
    cent_x, cent_y = ant_pos[0] + center_offset_x, ant_pos[1] + WWC_vertical_offset * (1 if dc_orien == 1 else -1)
    hw, hh = WWC_width / 2, WWC_height / 2
    return [(cent_x + dx * hw, cent_y + dy * hh) for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]


"""
Section 3: 
----------
Now we have the functions to construct the constraining rectangles, we need to make functions that check for collisions.
Once the optimisation process is underway, each antenna will get shifted around and so we will need to know whether the 
rectangles overlap or touch.
"""


def antennas_inside_bounds(array):
    inside_bounds = True
    for pos in array:
        if check_inner_circle(pos) == False:
            inside_bounds = False
            break
    return inside_bounds


def check_inner_circle(pos):
    # This is just for the antennas
    r = array_radius
    x, y = pos
    # The r here is used to define the centre
    return ((x - r) ** 2 + (y - r) ** 2) < (inner_array_radius - ant_radius) ** 2


def no_collisions_all_in_bounds(array, DCs):
    safe_state = rects_inside_bounds(array, DCs) and no_rect_collisions(array, DCs) and no_ant_ant_collisions(
        array) and antennas_inside_bounds(array)

    return safe_state


def no_ant_ant_collisions(array):
    """
    This functions simply finds all the antenna pair separations and checks whether or not the minimum separation
    is greater than the antenna diamter
    """
    try:
        min_seps = min(calc_separations(array))
    except:
        # This try-except clause is to prevent an error when building the intial array. There will be a time where there is only 1 antenna and therefore no min_sep
        min_seps = 4 * ant_radius
    if min_seps <= 2 * ant_radius:
        return False
    else:
        return True


def no_rect_collisions(array, DCs):
    """
    This function will check whether or not any of the SMP or PCB rectangles collide
    """
    # Generate the positions/coordinates of all the PCBs
    PCB_coords = []
    SMP_coords = []
    WWC_coords = []

    for ant_indx in range(len(array)):
        dc_orien = DCs[ant_indx]
        ant_pos = array[ant_indx]
        PCB = generate_PCB_rect(ant_pos, dc_orien)
        SMP = generate_SMP_rect(ant_pos, dc_orien)
        WWC = generate_WWC_rect(ant_pos, dc_orien)
        PCB_coords.append(PCB)
        SMP_coords.append(SMP)
        WWC_coords.append(WWC)

    # We only need to see if all the PCBs have any collisions with either other PCBs or any of the SMPs
    is_collision = False

    for PCB_indx1 in range(len(array)):

        # This loop will check to see if any of the PCBs collide with any other PCBs
        for PCB_indx2 in range(len(array)):
            if PCB_indx2 != PCB_indx1:
                # So as long it isn't the same one
                rect1, rect2 = PCB_coords[PCB_indx2], PCB_coords[PCB_indx1]
                collision = two_rect_intersect(rect1, rect2)
                if collision:
                    is_collision = True
                    break

        # This loop will check to see if any of the PCBs collide with any of the SMPs
        for SMP_indx in range(len(array)):
            if PCB_indx1 != SMP_indx:
                rect1, rect2 = PCB_coords[PCB_indx1], SMP_coords[SMP_indx]
                collision = two_rect_intersect(rect1, rect2)
                if collision:
                    is_collision = True
                    break

        # Check to see if any PCB collides with any WWC
        for WWC_indx in range(len(array)):
            if PCB_indx1 != WWC_indx:
                # So as long it isn't the same one
                rect1, rect2 = PCB_coords[PCB_indx1], WWC_coords[WWC_indx]
                collision = two_rect_intersect(rect1, rect2)
                if collision:
                    is_collision = True
                    break

    for SMP_indx1 in range(len(array)):
        for SMP_indx2 in range(len(array)):
            if SMP_indx1 != SMP_indx2:
                rect1, rect2 = SMP_coords[SMP_indx1], SMP_coords[SMP_indx2]
                collision = two_rect_intersect(rect1, rect2)
                if collision:
                    is_collision = True
                    break

        for WWC_indx in range(len(array)):
            if SMP_indx1 != WWC_indx:
                rect1, rect2 = SMP_coords[SMP_indx1], WWC_coords[WWC_indx]
                collision = two_rect_intersect(rect1, rect2)
                if collision:
                    is_collision = True
                    break

    for WWC_indx1 in range(len(array)):
        for WWC_indx2 in range(len(array)):
            if WWC_indx1 != WWC_indx2:
                rect1, rect2 = WWC_coords[WWC_indx1], WWC_coords[WWC_indx2]
                collision = two_rect_intersect(rect1, rect2)
                if collision:
                    is_collision = True
                    break

    return not is_collision


def point_inside_rect(point_pos, rect):
    R1, R2, R3, R4 = rect
    x, y = point_pos
    if R1[0] <= x <= R2[0] and R1[1] <= y <= R4[1]:
        return True
    else:
        return False


def two_rect_intersect(rect1, rect2):
    bottom_left1, bottom_right1, top_right1, top_left1 = rect1
    bottom_left2, bottom_right2, top_right2, top_left2 = rect2
    return not (top_right1[0] < bottom_left2[0] or bottom_left1[0] > top_right2[0] or top_right1[1] < bottom_left2[1] or
                bottom_left1[1] > top_right2[1])


def rects_inside_bounds(array, DCs):
    """
    This function will check whether or not any of the SMP or PCB rectangles lie outside the outer circular boundary
    """
    # Generate the positions/coordinates of all the PCBs
    PCB_coords = []
    SMP_coords = []
    WWC_coords = []
    collision = False
    for ant_indx in range(len(array)):
        dc_orien = DCs[ant_indx]
        ant_pos = array[ant_indx]
        PCB = generate_PCB_rect(ant_pos, dc_orien)
        SMP = generate_SMP_rect(ant_pos, dc_orien)
        WWC = generate_WWC_rect(ant_pos, dc_orien)

        PCB_coords.append(PCB)
        SMP_coords.append(SMP)
        WWC_coords.append(WWC)
        # Check the SMP
        for coord in SMP:
            if check_bounds(coord, array_radius, 0) == False:
                # Outside circle
                collision = True
                break

        # Check the PCB
        for coord in PCB:
            if check_bounds(coord, array_radius, 0) == False:
                # Outside circle
                collision = True
                break
        # Check the WWC
        for coord in WWC:
            if check_bounds(coord, array_radius, 0) == False:
                # Outside circle
                collision = True
                break
    return not (collision)


def check_bounds(pos, r, d):
    """
    This function checks whether a certain position lies inside a circle of centre (a,b) and radius r.
    d repressent the min distance the point can get to the circle
    """
    a, b = array_radius, array_radius
    x, y = pos
    return ((x - a) ** 2 + (y - b) ** 2) < (r - d) ** 2


"""
Section 4: 
----------
Functions for the optimisation process
"""


def switch_DC_orien(ant_num, score_array, array, DCs):
    # Flip the down converter's orientation
    array, DCs = flip_dc(ant_num, array, DCs)
    safe = no_collisions_all_in_bounds(array, DCs)

    if safe:
        new_score = score_array(array)
        array, DCs = flip_dc(ant_num, array, DCs)
        old_score = score_array(array)
        # If the score has improved then flip it back to its new position
        if new_score > old_score:
            array, DCs = flip_dc(ant_num, array, DCs)
    else:
        array, DCs = flip_dc(ant_num, array, DCs)

    return array, DCs


def nudge_antenna(array, DCs, ant_num, tries_per_ant, score_array):
    # Saving the current x and y coordinates withing the array of that antenna number
    x_old, y_old = array[ant_num]

    # Given the current setup, what is the score?
    old_score = score_array(array)

    for j in range(tries_per_ant):

        x_nudge = np.random.normal(0, array_radius / 5)
        y_nudge = np.random.normal(0, array_radius / 5)

        x_new, y_new = x_old + x_nudge, y_old + y_nudge
        array[ant_num] = (x_new, y_new)

        good_move = no_collisions_all_in_bounds(array, DCs)

        if good_move:
            new_score = score_array(array)
            if new_score < old_score:
                # Then it was worse than before soo move it back
                array[ant_num] = (x_old, y_old)
        else:
            # There was a collision or something was out of bounds
            array[ant_num] = (x_old, y_old)
    return array, DCs

def nudge_antenna_sim_aneal(array, DCs, ant_num, tries_per_ant, score_array):

    probability_of_accepting_wose_solution = 0.002 # 1 %
    # Saving the current x and y coordinates withing the array of that antenna number
    x_old, y_old = array[ant_num]

    # Given the current setup, what is the score?
    old_score = score_array(array)

    for j in range(tries_per_ant):

        x_nudge = np.random.normal(0, array_radius / 5)
        y_nudge = np.random.normal(0, array_radius / 5)

        x_new, y_new = x_old + x_nudge, y_old + y_nudge
        array[ant_num] = (x_new, y_new)

        good_move = no_collisions_all_in_bounds(array, DCs)

        if good_move:
            new_score = score_array(array)
            if new_score < old_score:
                # Then it was worse than before soo move it back
                if np.random.uniform(low=0,high=1)>probability_of_accepting_wose_solution:
                    array[ant_num] = (x_old, y_old)
                else:
                    print("Accepted worse solution")

        else:
            # There was a collision or something was out of bounds
            array[ant_num] = (x_old, y_old)
    return array, DCs


import time
from copy import deepcopy


def optimise_array_design(array, DCs, iterations, tries_per_ant, main_scoring_function):
    """
    Optimizes the design of an array based on a scoring function.

    Parameters:
        array: Initial array configuration
        DCs: Initial DC configuration
        iterations: Number of iterations to run the optimization
        tries_per_ant: Number of attempts to nudge each antenna
        main_scoring_function: Scoring function used to evaluate the design

    Returns:
        Tuple with the scores, the best array, the best DCs, all arrays and all DCs.
    """

    # Initial settings
    count = 0
    scores = []
    arrays = []
    all_DCs = []

    # Deep copying initial array and calculating its score
    the_best_array = array.copy()
    the_best_array_DCs = DCs.copy()
    best_array_score = main_scoring_function(array)

    # Main optimization loop
    for n in range(iterations):
        start_time = time.time()

        # Inner loop to go through each antenna and try to nudge it to a better location
        for x in range(len(array)):
            count += 1

            # Nudging antenna and switching DC orientation
            array, DCs = nudge_antenna(array, DCs, x, tries_per_ant, main_scoring_function)
            array, DCs = switch_DC_orien(x, main_scoring_function, array, DCs)

            # Score the new array configuration
            score = main_scoring_function(array)

            # If new score is better, update the best score and best configurations
            if score > best_array_score:
                the_best_array = array.copy()
                the_best_array_DCs = DCs.copy()
                best_array_score = score

            # Record the configuration and score
            all_DCs.append(deepcopy(DCs))
            arrays.append(deepcopy(array))
            scores.append(score)

        # Save the best configurations to CSV files
        save_array_to_csv(the_best_array, "best_array.csv")
        save_DCs(the_best_array_DCs, "best_array_DCs.csv")

        # Calculate and print time per iteration
        tpi = time.time() - start_time
        print("-------------------------------------")
        print(
            f"Iteration: {n + 1}. Trial Nudge No: {count}. Score: {round(score, 5)}. Time Per iteration: {round(tpi, 3)}s")

    return scores, the_best_array, the_best_array_DCs, arrays, all_DCs


def optimise_array_design_w_scoring_function_option(array, DCs, iterations, tries_per_ant,main_scoring_function,other_scoring_function):
    from copy import deepcopy

    count = 0
    scores1= []
    scores2 = []
    arrays = []
    all_DCs = []


    for n in range(iterations):
        start_time = time.time()
        # Looping through each antenna and maybe nudging it a bit into a new better location
        for x in range(len(array)):
            count += 1
            array, DCs = nudge_antenna(array, DCs, x, tries_per_ant, main_scoring_function)
            array, DCs = switch_DC_orien(x, main_scoring_function, array, DCs)
        score1 = main_scoring_function(array)
        score2 = other_scoring_function(array)
        all_DCs.append(deepcopy(DCs))
        arrays.append(deepcopy(array))
        scores1.append(score1)
        scores2.append(score2)

        save_array_to_csv(array, "current_array.csv")
        save_DCs(DCs, "current_array_DCs.csv")
        tpi = time.time() - start_time
        print("-------------------------------------")
        print(
            f"Iteration: {n + 1}. Trial Nudge No: {count}. Score1: {score1}. Score2: {score2}.Time Per iteration: {round(tpi, 3)}s")

    return scores1,scores2, arrays, all_DCs


def flip_dc(ant_num, array, DCs):
    # Save old coords
    x, y = array[ant_num]
    # Set new coords
    array[ant_num] = (x + (PCB_width - 2.6) * DCs[ant_num], y)
    # Change dc orientaion
    DCs[ant_num] *= -1
    return array, DCs


def carry_on_optimising(array, DCs, num_iterations,scoring_function):
    """
    This function should be used when you already have an array and you wish to optimise it further

    :param array: A list containing the xy coordinates of each antenna element
    :param DCs: A list containing 1s and -1s representing the down convertor PCBs facing up and down.
    :param num_iterations: The number of iteration steps you want the optimisation function to run for
    :return: a list of scores coresponding to each iteration step, the length of time it took to complete all iterations, a list of all arrays and DCs (downconvertor orientations)
    """
    # Then we can start moving them around and optimising their positions
    start = time.time()
    scores, array, DCs, arrays, all_DCs = optimise_array_design(array, DCs, num_iterations,
                                                                                        tries_per_ant,scoring_function)
    time_taken = time.time() - start
    print(f"Took {round(time_taken / 60, 2)} mins")
    score = scores[-1]

    return scores, time_taken, arrays, all_DCs


def create_optimised_array(num_elements, num_iterations,scoring_function):
    # First we have to setup the array with some random antennas. 50 is just the number of attmepts
    for i in range(50):
        # Lets try up to 50 times to create an array with num_elements on it
        array, DCs = build_array(num_elements)
        if len(array) == num_elements:
            # Then we've done it, created an array with num_elements on it
            break
        if len(array) != num_elements and i == 50:
            # Then it has not been able to fit all of the antenna on the board
            print(f"Failed to fit all {num_elements} antenna onto the array")
            return 0, 0, [], []
    # Then we can start moving them around and optimising their positions
    start = time.time()
    scores, the_best_array, the_best_array_DCs, arrays, all_DCs = optimise_array_design(array, DCs, num_iterations,
                                                                                        tries_per_ant,scoring_function)
    time_taken = time.time() - start
    print(f"Took {round(time_taken / 60, 2)} mins")
    score = scores[-1]

    return scores, time_taken, arrays, all_DCs





"""
Section 5: 
----------
Now we need to make some functions to generate the initial phased array arrangement 
"""

def build_array(num_ants):
    array = []
    DCs = np.array([random.choice([1, -1]) for c in range(num_ants)])

    centre_pos = (array_radius, array_radius)
    array.append(centre_pos)

    for trial_ant_indx in range(num_ants):

        # If it can't place it after 1,000 tries its probably not going to
        for trial_placement in range(100):

            saved_array = array.copy()
            x_pos, y_pos = np.random.uniform(0, array_radius * 2), np.random.uniform(0, array_radius * 2)
            array.append((x_pos, y_pos))

            try:
                if check_for_collisions:
                    # Then check for collisions
                    good_move = no_collisions_all_in_bounds(array, DCs)
                else:
                    # Antennas inside those circles?
                    if len(array)<=num_ants:
                        good_move = antennas_inside_bounds(array)
                    else:
                        good_move=False

                if good_move == False:
                    array = saved_array.copy()
                else:
                    if random.choice([1, 0]) == 1:
                        array, DCs = flip_dc(trial_ant_indx, array, DCs)
                        if not (no_collisions_all_in_bounds(array, DCs)):
                            array, DCs = flip_dc(trial_ant_indx, array, DCs)
                    break



            except:
                array = saved_array.copy()
                # print("Placed all antennas down")
                break

    return array, DCs


"""
Section 6: 
----------
The following functons are responsible for saving and loading in both the array designs and the PCB orientations
"""


def save_array_to_csv(array, filename):
    x_values = [array[i][0] for i in range(len(array))]
    y_values = [array[i][1] for i in range(len(array))]
    array_df = pd.DataFrame()
    array_df["x"] = x_values
    array_df["y"] = y_values

    array_df.to_csv(f"{filename}", index=False)


# For beamforming
def load_array(filename):
    array_df = pd.read_csv(filename)
    array = [(array_df["x"][i], array_df["y"][i]) for i in range(len(array_df))]
    return array


def load_DCs(filename):
    DCs_df = pd.read_csv(filename, names=["DC_Orien"])
    DCs = DCs_df["DC_Orien"]
    return DCs


def save_DCs(DCs, filename):
    DCs = list(DCs)

    np.savetxt(filename,
               DCs,
               delimiter=", ",
               fmt='% s')


def load_in_array(filename):
    # This is because in the optimisation process I have stupidly used units of cm not metres.

    # Change this for a different array radius:
    array_radius = 9.75 / 100
    global freq, speed
    df = pd.read_csv(filename)
    array = []
    for r in range(len(df)):
        array.append(Antenna(df["x"][r] / 100 - array_radius, df["y"][r] / 100 - array_radius, 0))
    # Make each antenna a source

    # Source properties:
    speed = 3e8

    # They all lie on Z=0
    sources = [Source(freq, speed, (array[i].x, array[i].y, 0)) for i in range(len(array))]
    return array, sources


"""
Section 7: 
----------
The next chunk of functions are responsible for calculating statistics and various other things about the phased array design
"""


def pair_finder(array):
    """
    This function just takes the array as an input, which is a list of tuples,
    or a list of x and y coordinates and returns all of the antenna pairing coordinates
    """

    pairs = []
    N = len(array)  # Or you could do the length of the array
    for i in range(N):
        # The first antanna will have antenna number 0 to match pythonic indices
        for j in range(N):
            if i < j:
                pair = (array[i], array[j])
                pairs.append(pair)
    return pairs


def mae(a, b):
    a = np.array(a)
    b = np.array(b)
    res = (np.abs(a - b)).mean()
    return res


def calc_separations(array):
    pairs = pair_finder(array)
    separations = []
    for pair in pairs:
        x1, y1 = pair[0][0], pair[0][1]
        x2, y2 = pair[1][0], pair[1][1]
        separation = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        separations.append(separation)
    return separations


def calc_separationsxy(array):
    pairs = pair_finder(array)
    separationsx = []
    separationsy = []
    for pair in pairs:
        x1, y1 = pair[0][0], pair[0][1]
        x2, y2 = pair[1][0], pair[1][1]
        separationx = np.abs(x2 - x1)
        separationsx.append(separationx)
        separationy = np.abs(y2 - y1)
        separationsy.append(separationy)
    return np.array(separationsx), np.array(separationsy)


def calc_pair_angles(array, pairs):
    angles = []
    for pair_indx in range(len(pairs)):
        pair = pairs[pair_indx]

        x1, x2 = pair[0][0], pair[1][0]
        y1, y2 = pair[0][1], pair[1][1]

        angle = math.atan2((y2 - y1), (x2 - x1))
        if angle >= np.pi:
            angle += -np.pi
        angles.append(angle)
    return angles


"""
Section 8: 
----------
In order for the optimisation program to run we need to be able to "score" each design it makes, that way we can decide whether or not to
keep it. The following functions deal with the beamforming and gain determination
Scoring functions:
"""
def score_array_angles_and_xy_separation(array):
    angles_score = score_array_angles(array)
    xysep_score = score_array_xy_sep(array)

    score = (angles_score+xysep_score)/2

    return score

def score_array_seps_mag(array):
    pairs = pair_finder(array)
    seps = calc_separations(array)
    seps.sort()

    # Perfect distribution:
    ideal_sep = np.linspace(ant_radius, inner_array_radius, len(pairs))
    error = np.mean((ideal_sep - seps) ** 2)

    return 1 / error


def score_array_xy_sep(array):
    pairs = pair_finder(array)
    separationsx, separationsy  = calc_separationsxy(array)
    separationsx.sort()
    separationsy.sort()


    # Perfect distribution:
    ideal_sep = np.linspace(ant_radius,inner_array_radius,len(pairs))
    errorx = np.mean((ideal_sep - separationsx) ** 2)
    errory = np.mean((ideal_sep - separationsy) ** 2)
    error = (errorx+errory)/2

    return 1/error

def score_array_angles(array):

    pairs = pair_finder(array)
    pair_angles = calc_pair_angles(array, pairs)
    pair_angles.sort()
    ideal_pair_distribution = np.linspace(-np.pi,np.pi,len(pair_angles))
    error = np.mean((ideal_pair_distribution-pair_angles)**2)

    # We want the error to be as low as possible so the score should be 1/error
    score = 1/error
    return score

def calc_gain(Im):
    center_image, outside_image = cut_out_center(Im, circle_gain_crop_radius)

    gain = max(center_image) / max(outside_image)
    return gain


def score_gain_fast(array):
    global res
    filename = "temp_array.csv"
    save_array_to_csv(array, filename)

    I = create_image(filename,res)

    gain = calc_gain(I)
    jiggle_amount = gain * tiny_jiggle_score / 100
    random_deviation = np.random.uniform(low=-jiggle_amount, high=jiggle_amount)

    return gain + random_deviation

def sidelobe_rejection_function_db(array):
    global res
    filename = "temp_array.csv"
    save_array_to_csv(array, filename)

    I = create_image(filename, res)

    gain = calc_gain(I)

    sidelobe_rejection = 10*np.log10(1/gain)

    jiggle_amount = gain * tiny_jiggle_score / 100
    random_deviation = np.random.uniform(low=-jiggle_amount, high=jiggle_amount)

    #score = -1*sidelobe_rejection + random_deviation
    return sidelobe_rejection,I,gain

def score_directivity(array):
    filename = "temp_array.csv"
    save_array_to_csv(array, filename)
    I = create_image(filename,res)
    I /= I.max()

    directivity = 10 * np.log10(1 / I.mean())
    return directivity
def score_SNR(array):
    filename = "temp_array.csv"
    save_array_to_csv(array, filename)

    I = create_image(filename, res)

    gain = calc_gain(I)

    sidelobe_rejection = 10 * np.log10(1 / gain)

    jiggle_amount = gain * tiny_jiggle_score / 100
    random_deviation = np.random.uniform(low=-jiggle_amount, high=jiggle_amount)

    score = -1*sidelobe_rejection + random_deviation
    return score


def create_AP(filename):
    # AP stands for antenna positions. We want them in cm
    df = pd.read_csv(filename)
    x = df["x"] / 100 - array_radius / 100
    y = df["y"] / 100 - array_radius / 100
    z = np.zeros(len(x))

    AP = []
    for i in range(len(x)):
        position_of_single_antenna = (x[i], y[i], z[i])
        AP.append(position_of_single_antenna)
    AP = np.array(AP)

    return AP


def create_planar_surface(width, Z_depth, res):
    X = np.linspace(-width/2, width/2, res)
    Y = np.linspace(-width/2, width/2, res)
    X_full = []
    Y_full = []
    Z_full = []
    for x in X:
        for y in Y:
            X_full.append(x)
            Y_full.append(y)
            Z_full.append(Z_depth)

    PP = []
    for i in range(len(X_full)):
        position_of_single_antenna = (X_full[i], Y_full[i], Z_full[i])
        PP.append(position_of_single_antenna)
    PP = np.array(PP)
    surface = (X_full, Y_full, X_full)
    return PP, surface

def cut_out_center(image, radius):
    """Cut out a circular region from the center of an image.

    Args:
        image (numpy.ndarray): A 2D array representing the image.
        radius (int): The radius of the circular region to cut out.

    Returns:
        tuple of numpy.ndarray: A pair of arrays containing the center and outside regions of the image.
    """
    # Get the shape of the image
    rows, cols = image.shape

    # Create a mask of all False values with the same shape as the image
    mask = np.zeros_like(image, dtype=bool)

    # Create a center point for the circular shape
    center = (rows // 2, cols // 2)

    # Use NumPy's meshgrid function to create a grid of x and y values
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Use the euclidean distance formula to calculate the distance from each point in the grid to the center point
    dist = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    # Create a mask where all points within the specified radius of the center point are True
    mask_inside = dist <= radius
    mask_outside = dist >= radius

    # Use the mask to extract the center and outside regions of the image
    center_image = image[mask_inside]
    outside_image = image[mask_outside]

    return center_image, outside_image

def outside_circle_flatness(array):


    filename = "temp_array.csv"
    save_array_to_csv(array, filename)

    I = create_image(filename, res)


    center_image, outside_image = cut_out_center(I, circle_gain_crop_radius)

    std_outside = np.std(outside_image)
    max_val = max(center_image)

    return (max_val/std_outside)


def create_interference_pattern(D,res):
    """
    D is a matrix with the separation values between each screen pixel and each antenna element
    """
    # number of time steps
    num_steps = 3

    # time step size
    dt = T / num_steps

    # initialize array to store intensity at each time step
    I = [np.sum(((1 / D) * np.cos(k * D - omega * i * dt)), axis=0) ** 2 for i in range(num_steps)]

    # calculate mean intensity over all time steps
    I = np.mean(I, axis=0)

    image = I.reshape(res, res)

    return image



def create_image_v2(AP,res,Z_depth,width):


    # PP is the matrix that contains nformation about the screen pixel positions. (X,Y,Z) are the coordinates of the screen pixels
    PP, (X, Y, Z) = create_planar_surface(width, Z_depth, res)

    # D is the matrix that contains information about the antenna-pixel separations and therfore the phase differences
    D = np.sqrt(np.sum((AP[:, np.newaxis, :] - PP[np.newaxis, :, :]) ** 2, axis=-1))

    I = create_interference_pattern(D,res)

    return I

def create_image(filename,res):
    # Matrix holding the antenna positions
    AP = create_AP(filename)

    # PP is the matrix that contains nformation about the screen pixel positions. (X,Y,Z) are the coordinates of the screen pixels
    PP, (X, Y, Z) = create_planar_surface(width, Z_depth, res)

    # D is the matrix that contains information about the antenna-pixel separations and therfore the phase differences
    D = np.sqrt(np.sum((AP[:, np.newaxis, :] - PP[np.newaxis, :, :]) ** 2, axis=-1))

    I = create_interference_pattern(D,res)

    return I


"""
Section 9: 
----------
Functions to help visualise the array designs
"""


def display_array(array):
    plt.figure(figsize=(10, 10), dpi=40)
    plt.title("Array Design", fontsize=20)
    df_temp = pd.DataFrame(array, columns=["x", "y"])
    sns.scatterplot(data=df_temp, x="x", y="y", s=200, marker="x", color="black")
    plt.xlabel("x (cm)", fontsize=20)
    plt.ylabel("y (cm)", fontsize=20)
    plt.xlim(0, 2 * array_radius)
    plt.ylim(0, 2 * array_radius)

    plt.show()


def plot_rectangles(rects, color, fill=False):
    # Plotting the rectangles
    for rect in rects:
        p1, p2, p3, p4 = rect
        plt.plot([p1[0], p2[0], p3[0], p4[0], p1[0]], [p1[1], p2[1], p3[1], p4[1], p1[1]], color=color, linewidth=1.5)
        if fill:
            plt.fill([p1[0], p2[0], p3[0], p4[0]], [p1[1], p2[1], p3[1], p4[1]], color=color)


def extract_phases(signals, main_frequency=10.36e6, delta_nu=0.2e6):
    fft_signals = fft(signals, axis=1)

    # Calculate the frequency bins
    sample_freq = 125e6
    n_samples = fft_signals.shape[1]
    freq_bins = fftfreq(n_samples, 1 / sample_freq)

    # Select a narrow band of frequency around the main frequency

    selected_freq_indices = np.where(
        (freq_bins >= main_frequency - delta_nu) & (freq_bins <= main_frequency + delta_nu))

    narrow_band_ffts = np.array([fft_signals[selected_freq_indices] for fft_signals in fft_signals])
    frequencies = freq_bins[selected_freq_indices]

    phases = np.angle(narrow_band_ffts)  # Shape: (9, 27)
    phases = np.array([phases[i][phases.shape[1] // 2] for i in range(len(phases))])
    return phases


def capture_interference_on_screen(filename, res):
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    I = create_image(filename, res)
    I /= I.max()
    I_db = 10*np.log10(I)
    sns.set_style("dark")

    # Create a meshgrid for the circle and the axes limits
    x = np.linspace(-width / 2, width / 2, res)
    y = np.linspace(-width / 2, width / 2, res)
    X, Y = np.meshgrid(x, y)

    """
    # Plot the image
    fig, ax = plt.subplots(figsize=(10, 10), dpi=80)
    im = ax.imshow(
        I,
        cmap="magma",
        extent=(-width / 2, width / 2, -width / 2, width / 2),
        #vmin=-30,  # Set the minimum value for the colorbar range
        #vmax=0,  # Set the maximum value for the colorbar range
    )

    ax.set_xlabel("Pixel Position (m)", fontsize=18)
    ax.set_ylabel("Pixel Position  (m)", fontsize=18)

    # Set up the colorbar to align with the height of the image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Power", fontsize=18)

    plt.show()
    """

    return I


def display_array_full(array, DCs):

    sns.set_style("dark")
    plt.figure(figsize=(10, 10), dpi=100)

    # Plotting the outer bounding circle
    thetas = np.linspace(0, 2 * np.pi, 50)
    x_values = array_radius * np.cos(thetas) + array_radius
    y_values = array_radius * np.sin(thetas) + array_radius
    plt.plot(x_values, y_values, color="red")
    # Plotting the inner bounding circle
    thetas = np.linspace(0, 2 * np.pi, 50)
    x_values = inner_array_radius * np.cos(thetas) + array_radius
    y_values = inner_array_radius * np.sin(thetas) + array_radius
    plt.plot(x_values, y_values, color="black")

    # Plotting the PCBs
    PCBs = [generate_PCB_rect(array[i], DCs[i]) for i in range(len(array))]
    plot_rectangles(PCBs, "black", fill=True)

    # Plotting the SMPs
    SMPs = [generate_SMP_rect(array[i], DCs[i]) for i in range(len(array))]
    plot_rectangles(SMPs, "green", fill=True)

    # Plotting the WWCs
    WWCs = [generate_WWC_rect(array[i], DCs[i]) for i in range(len(array))]
    plot_rectangles(WWCs, "purple", fill=True)

    # Plot the antennas
    x_positions = [array[i][0] for i in range(len(array))]
    y_positions = [array[i][1] for i in range(len(array))]
    plt.scatter(x_positions, y_positions, s=600,color="black")

    plt.xlabel("x (cm)", fontsize=19)
    plt.ylabel("y (cm)", fontsize=19)
    plt.xlim([0, array_radius * 2])
    plt.ylim([0, array_radius * 2])
    plt.show()

def visualise_scores(scores,title):

    df = pd.DataFrame()
    df["Design_Gain"] = 10 * np.log10(1/np.array(scores)) # Expressed as dB
    df["Antenna_Movement"] = np.array(range(1,len(scores)+1))
    sns.set_style("whitegrid")
    plt.figure(figsize=(10,4),dpi=100)
    plt.title(title,fontsize=18)
    plt.xlabel("Antenna Position Adjustment",fontsize=18)
    plt.ylabel("Design Gain ($dB$)",fontsize=18)
    sns.lineplot(data=df,x="Antenna_Movement", y="Design_Gain",color="black")
    plt.xscale("log") # Log scale for x axis

    plt.show()

"""
Section 10 - User defined parameters: 
----------
Now we need to set up and populate the physical system with our desired constraints. 
In reality we have a circular drum-like casing that holds a certain number of semi-circular (sinuous) antenna on the front pannel.
However, just behind each sinuous antenna, which we will assume is circular, there are various components we must consider.
Each antenna has a printed circuit board just behind it which will be modelled by a selection of rectangles.
Then there is the fact that the orientation of the PCB matters, so we must allow it to flip its orientation.
"""

"""
In this python file you must set all of the constaints, settings and parameters. 
"""



"""
Now you must set the parameters for the physical constraints. I.e the size of everything.
Below you can see various tolerances have been chosen depending on the component
"""

# Dimentions are all in cm
# PCB


PCB_width = (5.1 + (5.1) * 0.05)
PCB_height = (0.5 + (0.5) * 0.05)  # 5mm

# SMP
SMP_width = (2.25 + 0.05 * 2.25)
SMP_height = (1.1 + 0.05 * 1.1)  # 11mm

# WithWave Connectors
WWC_width = (2.5 + 2.5 * 0.05)
WWC_height = (1 + 1 * 0.05)
WWC_vertical_offset = 0

# Set the global variables
array_radius = 12  # This is the outer circle governing the PCB boundaries
inner_array_radius = 8  # This is the inner circle governing the actual antennas' boundaries
ant_radius = (0.7 + 0.7 * 0.05)

"""

# If you wish to have no constraints:
ant_radius = 0
WWC_height = 0
WWC_width = 0
WWC_vertical_offset = 0

SMP_height = 0
SMP_width = 0
PCB_height = 0
PCB_width = 0
# Set the global variables
array_radius = 12  # This is the outer circle governing the PCB boundaries
inner_array_radius = 8  # This is the inner circle governing the actual antennas' boundaries
ant_radius = 0




Settings for the optimisation process.
----------------------------------------
- Set scoring_func_alternate to true if you with to use an alternative scoring function every n iterations
- This is where you set the scoring function you wish to use
    - You can choose from:
        - score_array_angles_bow
        - score_gain_fast
- Choose the number of antennas you wish to create a design with
- iterations is the number of times the program will loop over the entire array
- tries_per_ant is the number of times the program will try to move each indivaidual antenna
to a better position each iteration before giving up and moving on
"""

# Optimisation Settings


iterations = 100 # You can leave this
tries_per_ant = 1
every_n_iterations = 400  # Use the other scoring function. This can be ignored as we are only using 1 function now
main_scoring_function = score_gain_fast
tiny_jiggle_score = 0.1  # This adds or subtracts a random percentage of the score onto the score..
# This allows for the program to keep a lower performing score and esacpe some local minima


"""
Parameters for the array's beamforming, and interference pattern generation:
"""
freq = 20e9
speed = 3e8
res = 200 # Resolution of the screen
width = 1 # 1m
Z_depth = 1 #  1m
circle_gain_crop_radius_m = 2*100e-3  # 90mm



# Ignore these:
circle_gain_crop_radius = circle_gain_crop_radius_m * (res / (width * 2))  # pixels
T = 1 / freq
omega = 2 * np.pi * freq
k = omega / speed
check_for_collisions = True