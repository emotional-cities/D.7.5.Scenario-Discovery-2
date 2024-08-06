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

def rf_fit(X, y):

    # Definir los parámetros que quieres ajustar y sus posibles valores
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None,10,20]
    }

    # Inicializar el clasificador Random Forest
    rf_clf = RandomForestRegressor(random_state=42)

    # Inicializar GridSearchCV
    grid_search = GridSearchCV(rf_clf, param_grid, cv=3)

    # Ajustar GridSearchCV para encontrar los mejores parámetros
    grid_search.fit(X, y)

    # Obtener el mejor modelo
    best_rf_model = grid_search.best_estimator_

    # Acceder a los mejores parámetros
    best_params = grid_search.best_params_
    print("Mejores parámetros:", best_params)


    return best_rf_model

def make_predictions_rf(best_rf_model, X_test):

    # Realizar predicciones con el mejor modelo
    predictions = best_rf_model.predict(X_test)

    return predictions


############################################################################################################

###### Execution code
import pandas as pd
with open('Data/X10_4.pickle', 'rb') as file:
    X = pickle.load(file)
with open('Data/Xtest40_4.pickle', 'rb') as file:
    X_test = pickle.load(file)

mode = 'free'

n_iter = 6 #idealy 6

#different paths for the policy that I want to apply
if mode == 'cheap':
    command = "./SimMobility_Medium ./simulation_maas_base_case_taz_cheap.xml ./simrun_MidTerm_maas_base_case.xml"
if mode == 'normal':
    command = "./SimMobility_Medium ./simulation_maas_base_case.xml ./simrun_MidTerm_maas_base_case.xml"
if mode == 'free':
    command = "./SimMobility_Medium ./simulation_maas_base_case_taz_free.xml ./simrun_MidTerm_maas_base_case.xml"


# outcome = []

# #Create a dataframe with the LHS initial dataset
# for i in range(X.shape[0]):
#     #compute the value of the new point given the model
#     model_data = ['dpb','dpt', 'nte', 'nto', 'nts','ntw', 'tme', 'tws', 'uw', 'dpb', 'dps', 'dpt', 'nte', 'nto', 'nts', 'ntw', 'tws', 'uw', 'dpb', 'dpb', 'dps', 'dpt', 'nte', 'nto', 'nts', 'ntw', 'uw', 'tws', 'tmw']
#     param_name_data = ['beta_lin_uncertainty', 'beta_lin_uncertainty','beta_lin_uncertainty','beta_lin_uncertainty', 'beta_lin_uncertainty', 'beta_lin_uncertainty','beta_lin_uncertainty','beta_lin_uncertainty','beta_lin_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_female_travel_uncertanty', 'uncertainty_income', 'uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income', 'uncertainty_income', 'uncertainty_income']
#     declaration_data = ['local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local']
#     value_data = [X[i,0],X[i,0],X[i,0], X[i,0], X[i,0], X[i,0],X[i,0], X[i,0], X[i,0], X[i,1], X[i,1], X[i,1], X[i,1], X[i,1],X[i,1],X[i,1],X[i,1],X[i,1], X[i,2], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3], X[i,3]]
#     data = {
#         'model': model_data,
#         'param_name': param_name_data,
#         'declaration': declaration_data,
#         'value': value_data
#     }

#     # Create a DataFrame from the dictionary
#     params = pd.DataFrame(data)
#     outcome.append(fn_simulation(params, command))

# X = np.array(X)
# y1 = np.array([outcomelist[0] for outcomelist in outcome])
# y2 = np.array([outcomelist[1] for outcomelist in outcome])


import pandas as pd
import pickle
with open('Y_train_1_PT_new'+ mode + '.pickle', 'rb') as handle:
    y1 = pickle.load(handle)
with open('Y_train_3_PT_new'+ mode + '.pickle', 'rb') as handle:
    y3 = pickle.load(handle)

#Compute metamodel
best_rf_model = rf_fit(X, y1)

X_new_labeled = X
y_new_labeled = y1
y_new_labeled_2 = y3


for j in range(n_iter):

    #Compute the posterior in many LHS points
    predictions = make_predictions_rf(best_rf_model, X_test)

    first_column_percentile = np.percentile(predictions, 20)

    #Compute the vulnerable cases
    Y= [1 if (predictions[i] < first_column_percentile) else 0 for i in range(len(predictions))]
    Y= np.array(Y).astype(float)
    print("Number of vulnerable scenarios: ", sum(Y), 'out of', len(Y))

   #perform PRIM in the posterior LHS
    p = prim.Prim(X_test, Y, threshold=0.3, threshold_type=">")
    box = p.find_box()

    df = box.limits

    #create an array to store the contrain dimesions
    dimension = np.zeros([(X.shape[-1]),2])
    dimension[:,1] = 1

    #store the restricted dimension from PRIM for each dimension
    for i in range(len(df)):
        dimension[int(df.index[i]),0] = df.iloc[i,0] #minumum
        dimension[int(df.index[i]),1] = df.iloc[i,1] #maximum

    #sample a random point within the restricted dimensions
    new_possible_points = []
    new_X_point = np.zeros(X.shape[-1])
    for i in range(X.shape[-1]):
        new_X_point = np.zeros(X.shape[-1])
        new_X_point[i] = random.uniform(dimension[i,0],dimension[i,1])
        for e in range(X.shape[-1]):
            if e != i: 
                new_X_point[e] = random.choices([dimension[e,0],dimension[e,1]], k=1)[0]
    new_possible_points.append(new_X_point)

    #choice a point from the possible ones
    new_X_point = random.choices(new_possible_points, k=1)[0]
    print('New point: ', new_X_point)

    #compute the value of the new point given the model
    #compute the value of the new point given the model
    model_data = ['dpb','dpt', 'nte', 'nto', 'nts','ntw', 'tme', 'tws', 'uw', 'dpb', 'dps', 'dpt', 'nte', 'nto', 'nts', 'ntw', 'tws', 'uw', 'dpb', 'dpb', 'dps', 'dpt', 'nte', 'nto', 'nts', 'ntw', 'uw', 'tws', 'tmw']
    param_name_data = ['beta_lin_uncertainty', 'beta_lin_uncertainty','beta_lin_uncertainty','beta_lin_uncertainty', 'beta_lin_uncertainty', 'beta_lin_uncertainty','beta_lin_uncertainty','beta_lin_uncertainty','beta_lin_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_age_uncertainty', 'beta_female_travel_uncertanty', 'uncertainty_income', 'uncertainty_income', 'uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income','uncertainty_income', 'uncertainty_income']
    declaration_data = ['local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local', 'local']
    value_data = [new_X_point[0],new_X_point[0],new_X_point[0], new_X_point[0], new_X_point[0], new_X_point[0],new_X_point[0], new_X_point[0], new_X_point[0], new_X_point[1], new_X_point[1], new_X_point[1], new_X_point[1],new_X_point[1],new_X_point[1],new_X_point[1],new_X_point[1],new_X_point[1], new_X_point[2], new_X_point[3], new_X_point[3], new_X_point[3], new_X_point[3], new_X_point[3], new_X_point[3], new_X_point[3], new_X_point[3], new_X_point[3], new_X_point[3]]
    data = {
        'model': model_data,
        'param_name': param_name_data,
        'declaration': declaration_data,
        'value': value_data
    }

    ## Create a DataFrame from the dictionary
    params = pd.DataFrame(data)
    outcome = fn_simulation(params, command)

    outcome1 = np.array([outcome[0]])
    outcome2 = np.array([outcome[2]])

    #Add point to the X train samples
    X_new_labeled =  np.vstack((X_new_labeled, new_X_point))
    y_new_labeled = np.concatenate((y_new_labeled, outcome1))
    y_new_labeled_2 = np.concatenate((y_new_labeled_2, outcome2))

    print('X_new_labeled size: ', X_new_labeled.shape)
    print('y_new_labeled size: ', y_new_labeled.shape)

    j = j + 1
    print("Iteration: ", j)

    #every 3 times I compute a new metamodel
    if (j % 3 == 0):
        print("Computing a new metamodel")
        best_rf_model = rf_fit(X_new_labeled, y_new_labeled)
    j = j - 1

#Get the final value of the posterior for the LHS samples
predictions = make_predictions_rf(best_rf_model, X_test)
print('Final predictions made')


#Save the list with the results
with open('Prediction_test_PRIM_PT_1_dim_UNC_sampling_2_dim_1' + mode + '.pickle', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
#Save the list with the results
with open('X_final_PRIM_PT_1_dim_UNC_sampling_2_dim_1' + mode + '.pickle', 'wb') as handle:
    pickle.dump(X_new_labeled, handle, protocol=pickle.HIGHEST_PROTOCOL)
#Save the list with the results
with open('Y_final_PRIM_PT_1_dim_UNC_sampling_2_dim_1'+ mode + '.pickle', 'wb') as handle:
    pickle.dump(y_new_labeled, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('Y_final_total_women_trips_dim_2_sampling_2_dim_1'+ mode + '.pickle', 'wb') as handle:
    pickle.dump(y_new_labeled_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
