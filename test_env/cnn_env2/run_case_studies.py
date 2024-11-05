#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" Payam Ghassemi | 02/27/2020 """
import os.path

import numpy as np
from BayesSwarm.simulator import Simulator
from BayesSwarm.types import SimulationConfigs
import csv
import scipy.io


def main():
    debug = True
    case_studies = [805]
    for cs in case_studies:
        case_study = cs
        decision_making_mode = "bayes-swarm"  # "bayes-swarm"
        simulation_mode = " "  # Options: "pybullet", " "
        if simulation_mode == "pybullet":
            simulation_configs = SimulationConfigs(
                simulation_mode="pybullet",
                environment="mountain-1",
                texture="source",
                robot_type="uav")
        else:
            simulation_configs = SimulationConfigs()
        is_scout_team = False
        start_locations = None
        observation_frequency = 1
        time_profiling_enable = False
        optimizers = [None, None]

        if case_study == 805:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 5
            source_id = 8
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 820:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 20
            source_id = 8
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 835:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 35
            source_id = 8
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 850:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 50
            source_id = 8
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 405:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 5
            source_id = 4
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 420:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 20
            source_id = 4
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 435:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 35
            source_id = 4
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 450:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 50
            source_id = 4
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 205:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 5
            source_id = 2
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 220:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 20
            source_id = 2
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 235:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 35
            source_id = 2
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"
        elif case_study == 250:  # Time profiling of Case 8 with large team
            time_profiling_enable = False
            n_robots = 50
            source_id = 2
            filtering_mode = "none"  # "none"
            enable_full_observation = True  # True
            bayes_swarm_mode = "local-penalty"

        else:
            raise "Invalid Case Study!"

        arr_req = []

        if not os.path.exists(f"cnn/{case_study}"):
            os.makedirs(f"cnn/{case_study}")


        # Open the CSV file in write mode

        for mission in range(5):

            sim = Simulator(n_robots=n_robots, source_id=source_id, start_locations=start_locations,
                            decision_making_mode=decision_making_mode, bayes_swarm_mode=bayes_swarm_mode,
                            filtering_mode=filtering_mode, observation_frequency=observation_frequency,
                            optimizers=optimizers, enable_full_observation=enable_full_observation,
                            is_scout_team=is_scout_team,
                            debug=debug, time_profiling_enable=time_profiling_enable,
                            simulation_configs=simulation_configs, mission=mission)

            print(f"MISSION_{mission}_STARTS")
            sim.run()
            time_ac = sim.mission_time
            arr_req.append(time_ac)
            decision_time = sim.decision_time
            with open(f"cnn/{case_study}/decision_time_{mission}.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                for tm in decision_time:
                    writer.writerow([tm])

            with open(f"cnn/mission_time_{case_study}.csv", 'w', newline='') as file:
                writer = csv.writer(file)
                for tm in arr_req:
                    writer.writerow([tm])

            with open(f"cnn/decision_time_{case_study}.csv", 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(decision_time)


            print(arr_req)
            # print(dec_time)
            print("MISSION_ENDS")


if __name__ == "__main__":
    # execute only if run as a script
    main()
