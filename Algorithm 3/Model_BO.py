import os
import numpy as np
import pandas as pd
import subprocess
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import prim
import random

def fn_simulation_config(params):
    """
    Update the simulation configuration using given parameters.

    Args:
        params (pd.DataFrame or list of dict): Input parameters to update the simulation configuration.

    Returns:
        list: List of statuses indicating the success or failure of each parameter update.
    """

    print("fn_simulation_config")

    # Initialize a list to store the status of each parameter update
    status = []

    # Convert the input params DataFrame into a list of dictionaries
    params = params.to_dict("records")

    # Iterate over each parameter and update the model configuration using a helper function
    for p in params:
        # The function fn_param_update is presumed to apply the parameter to the model 
        # and return a status indicating success or failure
        status.append(fn_param_update(p['model'], p['param_name'], p['declaration'], p['value']))

    # Return the list of status codes
    return status


def fn_param_update(model, param_name, declaration, value):
    """
    Update the specified parameter's value in the given lua model file.

    Args:
        model (str): The name of the lua model file (without the .lua extension).
        param_name (str): The name of the parameter to be updated.
        declaration (str): The declaration associated with the parameter (e.g., local).
        param_name_left_hand (str): Left-hand side representation of the parameter (likely deprecated or unused here).
        value (any): The new value to be assigned to the parameter.

    Note:
        This function assumes that the parameter declaration in the lua file is in the format:
        <declaration> <param_name> = <some_value>
        And it will replace <some_value> with the provided value.
    """
    
    # Define the directory path where the lua files are located
    lua_folder_path = "/home/ltola/simmobility/CPE/scripts/lua/mid/VC_Lorena"
    
    # Construct the full path to the desired lua file based on the given model name
    lua_file_path = os.path.join(lua_folder_path, f'{model}.lua')

    # Open the lua file and read its contents
    with open(lua_file_path, 'r') as file:
        filedata = file.read()

    # Search for the line that contains the target parameter name
    param_with_value = next((line for line in filedata.split('\n') if param_name in line), None)
    
    # If the parameter is found in the file, replace its value with the new one
    if param_with_value:
        filedata = filedata.replace(param_with_value, f'{declaration} {param_name} = {value}')

    # Write the modified content back to the lua file
    with open(lua_file_path, 'w') as file:
        file.write(filedata)


def fn_simulation_call(command):
    """
    Execute a simulation process in a specific directory and check for errors.

    Args:
        setting_obj (dict): A dictionary containing various settings. This argument is currently unused in the function.

    Returns:
        int: A return code indicating success (0) or failure (non-zero).
        
    Note:
        The function currently assumes a fixed simulation path and command.
        In case of an error, the function will raise an exception with the simulation's error messages.
    """
    
    print("fn_simulation_call")

    # Define the path to the simulation and the command to run it
    simulation_path = "/home/ltola/simmobility/CPE/"
    command = command

    # Execute the simulation command in the specified directory
    with subprocess.Popen(command, cwd=simulation_path, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            # Wait for the process to complete and capture any output (stdout and stderr)
            stdout, stderr = process.communicate()
            
           #  Uncomment the below lines if you want to print the simulation output (useful for debugging)
            print(f"Simulation Stdout:\n{stdout.decode('utf-8')}")
            print(f"Simulation Stderr:\n{stderr.decode('utf-8')}")
            
            # If the simulation returns a non-zero exit code, it means an error has occurred
            if process.returncode != 0:
                raise RuntimeError("Simulation terminated with an error!")
        except Exception as e:
            # If there's any issue in the above process, capture the error and terminate the simulation process
            print(f"Original Error: {e}")
            process.kill()
            print("Simulation Error")
            # Raise the captured error to the calling function or user
            raise

    # Return 0 to indicate a successful function execution
    return 0

def fn_process_activities():
    """
    Process activity schedules and return calculated output statistics.

    The function reads an activity schedule CSV file and computes output statistics
    based on the activity data. It utilizes the `fn_output_od_mode_balance` function
    to compute the desired outcomes.

    Returns:
        dict: A dictionary containing calculated output statistics.
        
    Note:
        This function assumes a global variable `shared_env` that contains 
        the settings, including the path to the activity schedule CSV file.
    """
    
    print("fn_process_activities")
    ### I change this to account for the accesibility
    # Define the expected column names for logsum CSV file
    # Define the expected column names for the activity schedule CSV file
    columns = ["person_id", "tour_no", "tour_type", "stop_no", "stop_type", "stop_location", "stop_zone", "stop_mode", "primary_stop", "arrival_time", "departure_time", "prev_stop_location", "prev_stop_zone", "prev_stop_departure_time", "pid"]

    # Read the specified CSV file into a pandas DataFrame using the defined column names
    activity = pd.read_csv("/home/ltola/simmobility/CPE/activity_schedule", names=columns, usecols=range(len(columns)))
    activity['person_id'] = activity['person_id'].str[:-2]
    activity['person_id'] = activity['person_id'].astype(int)
    # Calculate the output statistics
    #logsums["sum"] = logsums['work'] + logsums['education'] + logsums['shop'] + logsums['other'] + logsums['dp_tour'] # + logsums['dp_stop']
    #count_other = logsums[['person_id', 'sum']]

    # Specify the file path
    file_path = 'Data/VC_population.csv'

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path,delimiter = ";")
    df = df[['id', 'gender_id']]
    merged_df = pd.merge(activity, df, left_on='person_id', right_on='id', how='left')

    total_trips_women = len(merged_df[(merged_df['gender_id'] == 2)])
    total_trips_men = len(merged_df[(merged_df['gender_id'] == 1)])

    # Calcular el numero de shopping trips that women do on average
    PT_1_women = len(merged_df[(merged_df['gender_id'] == 2) & (merged_df['stop_mode'] == 'PT')])
    PT_2_women = len(merged_df[(merged_df['gender_id'] ==2) & (merged_df['stop_mode'] == 'PT_MOD_Pool')])
    PT_3_women = len(merged_df[(merged_df['gender_id'] ==2) & (merged_df['stop_mode'] == 'PT_Car')])

    sum_women_PT = PT_1_women

    # Calcular el numero de shopping trips that women do on average
    PT_1_men = len(merged_df[(merged_df['gender_id'] == 1) & (merged_df['stop_mode'] == 'PT')])
    #PT_2_men = len(merged_df[(merged_df['gender_id'] ==1) & (merged_df['stop_mode'] == 'PT_MOD_Pool')])
    #PT_3_men = len(merged_df[(merged_df['gender_id'] ==1) & (merged_df['stop_mode'] == 'PT_Car')])

    sum_men_PT = PT_1_men

    return [sum_women_PT, sum_men_PT, total_trips_women, total_trips_men]

def fn_simulation(params, command): #first function to call
    """
    Simulate a model or system using given parameters.

    Args:
        params (pd.DataFrame or dict): Input parameters for the simulation.

    Returns:
        dict: A dictionary containing configuration output, inadequacy value
    """

    print("fn_simulation")

    # Update the simulation configuration files with the given parameters
    param_config_output = fn_simulation_config(params)

    # Execute the actual simulation process, using a predefined global object "setting_obj"
    fn_simulation_call(command)

    print("The model has been run successfully")
    # Process the results of the simulation to get the activity outcomes
    outcome = fn_process_activities()

    # Prepare the inadequacy dictionary based on the outcome of activities
    #inadequacy = {"value": outcome}

    # Return a dictionary that includes configuration output, inadequacy values, and residuals
    return outcome

###############

def model(x1,x2,x3,x4): #first function to call

    mode = 'free'

    # with open('Y_train_1_new_' + str(zone) + '.pickle', 'rb') as file:
    #     y1 = pickle.load(file)
    # with open('Y_train_2_new_' + str(zone) + '.pickle', 'rb') as file:
    #      y2 = pickle.load(file)

    #different paths for the policy that I want to apply
    if mode == 'cheap':
        command = "./SimMobility_Medium ./simulation_maas_base_case_taz_cheap.xml ./simrun_MidTerm_maas_base_case.xml"
    if mode == 'normal':
        command = "./SimMobility_Medium ./simulation_maas_base_case.xml ./simrun_MidTerm_maas_base_case.xml"
    if mode == 'free':
        command = "./SimMobility_Medium ./simulation_maas_base_case_taz_free.xml ./simrun_MidTerm_maas_base_case.xml"


    outcome = []

    #compute the value of the new point given the model
    model_data = ['dpb','dpt', 'nte', 'nto', 'nts','ntw', 'tme', 'tws', 'uw', 'dpb', 'dps', 'dpt', 'nte', 'nto', 'nts', 'ntw', 'tws', 'uw', 'dpb', 'dpb', 'dps', 'dpt', 'nte', 'nto', 'nts', 'ntw', 'uw', 'tws', 'tmw']
    param_name_data = ['beta_lin_uncertainty', 'beta_lin_uncertainty','beta_lin_uncertainty','beta_lin_uncertainty', 'beta_lin_uncertainty', 'beta_lin_uncertainty','beta_lin_uncertainty','beta_lin_uncertainty','beta_lin_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_female_travel_uncertanty', 'uncertainty_income', 'uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income', 'uncertainty_income', 'uncertainty_income']
    declaration_data = ['local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local']
    value_data = [x1,x1,x1, x1, x1, x1, x1, x1, x1, x2, x2, x2, x2, x2,x2,x2,x2,x2, x3, x4, x4, x4, x4, x4, x4, x4, x4, x4, x4]
    data = {
        'model': model_data,
        'param_name': param_name_data,
        'declaration': declaration_data,
        'value': value_data
    }

    # Create a DataFrame from the dictionary
    params = pd.DataFrame(data)
    outcome = fn_simulation(params, command)

    y2 = outcome[1]

    return -y2