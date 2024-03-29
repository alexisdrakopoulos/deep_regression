"""
Contains utilities to be used, primarily for file management with s3

upload_to_s3 - uploads a single file to s3 bucket
download_from_s3 - downloads a single file from s3 bucket
download_all_files - downloads all files in movelog.txt file from s3
check_files - reads movelog.txt from s3 to check which files have been written
get_data - builds a list of all files to backup to s3
backup_files - uploads a list of files to s3 bucket
convert_to_categories - converts array of floating point values to bins
evaluate_models - uses prediction file to evaluate performance and writes to csv
regression_evaluations - simple evaluation metric calculations
predictions_to_csv - builds a csv file of all the prediction evaluations with log data
download_ising_data - downloads ising data numpy files into data/ising directory from s3

Args to pass to sys:
    download - Downloads all models/logs written to s3 that are in movelog.txt
    upload - Backs up all models/logs to s3 that are in movelog.txt
    predictions - Runs predictions to create csv file
    download_ising_files - downloads ising files to data/ising
"""

from sys import argv
import os
import subprocess
from pathlib import Path
from warnings import warn

if argv[1] in ("upload", "download", "download_all"):
    from multiprocessing import Pool

elif argv[1] in ("predictions", "write_model_details"):
    import numpy as np
    from sklearn.metrics import accuracy_score
    from keras.utils import to_categorical
    import csv


def upload_to_s3(path):
    """
    Moves a file from specified path to s3 bucket bucketname

    Inputs:
        path - string specifying path of file

    """

    bucket = "s3://bucketname"
    bucket_path = bucket + "/" + path.split("/")[-1]
    subprocess.run(f"aws s3 cp {path} {bucket_path}".split(" "))


def download_from_s3(path):
    """
    Moves a file from s3 bucket bucketname back to specified path

    Inputs:
        path - string specifying path of file (location to be written to)

    """

    bucket = "s3://bucketname"
    bucket_path = bucket + "/" + path.split("/")[-1]
    subprocess.run(f"aws s3 cp {bucket_path} {path}".split(" "))



def download_all_files(download_models=True, max_iter=53):

    core_path = "/dev/shm/deep_regression/"

    # Checking all the directories exist
    directories = ["data", "data/logs", "data/logs/model_logs",
                   "data/logs/training_logs", "data/models", "data/predictions"]

    for el in directories:
        if not os.path.exists(el):
            os.mkdir(el)

    # We need more info to parse the models and prediction names,
    # we will get that info by parsing the model_logs
    file_types = {"models":[],
                  "model_logs":[f"{core_path}data/logs/model_logs/log_{i}.txt" for i in range(max_iter)],
                  "training_logs":[f"{core_path}data/logs/training_logs/training_{i}.csv" for i in range(max_iter)],
                  "loss_histories":[f"{core_path}data/logs/loss_histories/{i}loss_histories.npy" for i in range(max_iter)],
                  "predictions":[]}

    files = file_types["model_logs"] + file_types["training_logs"] + file_types["loss_histories"]

    print("Downloading model logs, training logs and loss histories")
    with Pool() as p:
        print(p.map(download_from_s3, files))

    # now parsing the model logs to get the info we need
    for i in range(1, 53):

        log_lines = []
        with open(f"data/logs/model_logs/log_{i}.txt") as file: 
            for line in file:
                log_lines.append(line.split(": ")[-1].replace("\n", "").strip())

        model_type = log_lines[2]

        # now writing model + prediction name
        if model_type == "regression":
            model_type = "reg"
        else:
            model_type = "clas"

        file_types["models"].append(f"{core_path}data/models/bmodel_{i}_{model_type}.h5")
        file_types["models"].append(f"{core_path}data/models/fmodel_{i}_{model_type}.h5")
        file_types["predictions"].append(f"{core_path}data/predictions/bmodel_{i}_{model_type}_predictions.npy")
        file_types["predictions"].append(f"{core_path}data/predictions/fmodel_{i}_{model_type}_predictions.npy")

    print("Downloading model logs, training logs and loss histories")
    files = file_types["models"] + file_types["predictions"]
    with Pool() as p:
        print(p.map(download_from_s3, files))


def download_last_files(download_models=True):
    """
    downloads all files currently in s3 bucket bucketname
    """

    # First check the directory structure is correct
    directories = ["data", "data/logs", "data/logs/model_logs",
                   "data/logs/training_logs", "data/models", "data/predictions"]

    for el in directories:
        if not os.path.exists(el):
            os.mkdir(el)

    # list of files written to s3
    files = check_files()
    files = ["/" + i for i in files]

    new_files = []
    if download_models is False:
        for file in files:
            if "h5" not in file:
                new_files.append(file)
    files = new_files

    with Pool() as p:
        print(p.map(download_from_s3, files))


def check_files():
    """
    Pulls movelog.txt file from s3 bucket bucketname.
    This will fail if the bucket or the movelog.txt file do not exist.

    Outputs:
        files - list of files (full absolute path) written to movelog.txt
    """

    subprocess.run(f"aws s3 cp s3://bucketname/movelog.txt data/logs/movelog.txt".split(" "))

    files = []
    with open("data/logs/movelog.txt") as file: 
        for line in file: 
            files.append(line.replace("\n", ""))

    return files


def get_data():
    """
    Parses the directories data/models data/logs/model_logs and data/logs/training_logs
    in order to find all files that should be written to s3.

    Outputs:
        data - a list containing absolute paths to all the files found
    """

    print("Getting all data")
    data = []
    data_size = 0

    # Getting a list of models
    p = Path(".") / "data" / "models"
    for i in p.glob("*.h5"):
        data_size += i.stat().st_size
        data.append(str(i.absolute()))
    print(f"Total of {data_size/1E9} GB")

    # Getting a list of all model logs
    p = Path(".") / "data" / "logs" / "model_logs"
    for i in p.glob("*.txt"):
        data.append(str(i.absolute()))

    # Getting a list of all training logs
    p = Path(".") / "data" / "logs" / "training_logs"
    for i in p.glob("*.csv"):
        data.append(str(i.absolute()))

    # Getting a list of all loss histories
    p = Path(".") / "data" / "logs" / "loss_histories"
    for i in p.glob("*.npy"):
        data.append(str(i.absolute()))

    # Get all predictions
    p = Path(".") / "data" / "predictions"
    for i in p.glob("*.npy"):
        data.append(str(i.absolute()))

    return data


def backup_files():
    """
    Uploads a list of files to s3 bucket bucketname
    """

    # build a list of all files in directories to be backed up
    data = get_data()

    # check if the movelog.txt file exists in s3 bucket
    try:
        written_files = check_files()
    except FileNotFoundError:
        written_files = []


    # Find only files that have not been written before
    data_prunned = []
    for el in data:
        if el not in written_files:
            data_prunned.append(el)
        else:
            print("it's in here")

    print("\nWARNING: Are you sure you want to move all these files to S3?")
    answer = input("Y/N: ")

    # upload via multiprocessing to s3 for speed
    if answer.lower() in ("y", "yes"):
        with Pool() as p:
            print(p.map(upload_to_s3, data))

    # write text file with all the data and upload it to s3
    with open("movelog.txt", "w+") as output:
        for i in data_prunned:
            output.write(i + "\n")

    upload_to_s3("movelog.txt")


def convert_to_categories(values, bins):
    """
    Converts the continuous labels into evenly spread bins,
    uses the numpy linspace functionality.
    Can be slow, does around 500,000 iters/second.

    Inputs:
        values - numpy array of continuous labels (1D)
        bins - int value of number of bins to pass to numpy linspace

    Outputs:
        cats - the new 1D numpy array of categories from 0 to n in original order
    """

    vals = np.linspace(1, 4, bins)
    cats = np.zeros(len(values))

    print(f"Converting all labels to {bins} categories.")
    for idi, i in enumerate(values):
        for idj, j in enumerate(vals):
            if i < j:
                cats[idi] = idj
                break

    return cats


def evaluate_models(model_iteration, model_type, predicted, ground_truth, model_result):
    """
    Long function that uses predicted and ground truth numpy arrays to compute evaluations.
    Writes to csv file.

    Inputs:
        model_iteration - int specifying which model iteration is being evaluated
        model_type - str "regression" or "classification"
        predicted - numpy array of predicted values
        ground_truth - numpy array of true values
        model_result - str "best" or "final"
    """

    if model_type not in ("regression", "classification"):
        raise ValueError("model_type needs to be regression or classification")

    # Perform the evaluations
    if model_type == "regression":

        # Initialize the bool arrays for regression
        low = ground_truth < 2.27
        high = ground_truth > 2.27
        close = (ground_truth > 2) & (ground_truth < 2.54)

        predicted = np.squeeze(predicted)
        vals = regression_evaluations(predicted, ground_truth)
        low_vals = regression_evaluations(predicted[low], ground_truth[low])
        high_vals = regression_evaluations(predicted[high], ground_truth[high])
        close_vals = regression_evaluations(predicted[close], ground_truth[close])

    elif model_type == "classification":

        # Initialize the bool array for classifier
        low = ground_truth <= 4
        high = ground_truth > 4
        close = (ground_truth >= 3) & (ground_truth <= 5)

        ground_truth = ground_truth.astype(int)
        predicted = np.argmax(predicted, axis=1)
        vals = accuracy_score(predicted, ground_truth)
        low_vals = accuracy_score(predicted[low], ground_truth[low])
        high_vals = accuracy_score(predicted[high], ground_truth[high])
        close_vals = accuracy_score(predicted[close], ground_truth[close])

    field_names = ["Iteration", "Type", "Model", "MAPE", "MPE", "MSE", "MAPE Low", "MPE Low",
                   "MSE Low", "MAPE High", "MPE High", "MSE High", "MAPE Close",
                   "MPE Close", "MSE Close", "Accuracy", "Accuracy Low",
                   "Accuracy High", "Accuracy Close", "Date",
                   "Activation", "Batch Size", "Dropout", "Batchnorm",
                   "Batchnorm Order", "Learning Rate", "Sparse"]

    log_lines = []
    with open(f"data/logs/model_logs/log_{model_iteration}.txt") as file: 
        for line in file:
            log_lines.append(line.split(": ")[-1].replace("\n", "").strip())

    write_header = True
    if os.path.exists("model_evaluations.csv") is True:
            write_header = False

    with open('model_evaluations.csv', 'a+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        if write_header:
            writer.writeheader()

        if model_type == "regression":
            writer.writerow({"Iteration": model_iteration,
                             "Type": "reg",
                             "Model": model_result,
                             "MAPE": vals[1],
                             "MPE": vals[0],
                             "MSE": vals[2],
                             "MAPE Low": low_vals[1],
                             "MPE Low": low_vals[0],
                             "MSE Low": low_vals[2],
                             "MAPE High": high_vals[1],
                             "MPE High": high_vals[0],
                             "MSE High": high_vals[2],
                             "MAPE Close": close_vals[1],
                             "MPE Close": close_vals[0],
                             "MSE Close": close_vals[2],
                             "Accuracy": "",
                             "Accuracy Low": "",
                             "Accuracy High": "",
                             "Accuracy Close": "",
                             "Date": log_lines[0],
                             "Activation": log_lines[3],
                             "Batch Size": log_lines[4],
                             "Dropout": log_lines[5],
                             "Batchnorm": log_lines[6],
                             "Batchnorm Order": log_lines[7],
                             "Learning Rate": "%f" % eval(log_lines[9]),
                             "Sparse": log_lines[10]})

        elif model_type == "classification":
            writer.writerow({"Iteration": model_iteration,
                             "Type": "clas",
                             "Model": model_result,
                             "MAPE": "",
                             "MPE": "",
                             "MSE": "",
                             "MAPE Low": "",
                             "MPE Low": "",
                             "MSE Low": "",
                             "MAPE High": "",
                             "MPE High": "",
                             "MSE High": "",
                             "MAPE Close": "",
                             "MPE Close": "",
                             "MSE Close": "",
                             "Accuracy": vals,
                             "Accuracy Low": low_vals,
                             "Accuracy High": high_vals,
                             "Accuracy Close": close_vals,
                             "Date": log_lines[0],
                             "Activation": log_lines[3],
                             "Batch Size": log_lines[4],
                             "Dropout": log_lines[5],
                             "Batchnorm": log_lines[6],
                             "Batchnorm Order": log_lines[7],
                             "Learning Rate": "%f" % eval(log_lines[9]),
                             "Sparse": log_lines[10]})


def regression_evaluations(predicted, ground_truth):
    """
    computes some evaluation metrics such as % error, MAPE, MAE

    Inputs:
        predicted - an array of predicted values
        ground_truth - an array of the true values

    Outputs:
        tuple of 3 measurements mean % error, MAPE, MAE
    """

    error = predicted - ground_truth
    percentage_error = error / ground_truth

    mean_percentage_error = np.mean(percentage_error) * 100
    mean_abs_percentage_error = np.mean(np.abs(percentage_error)) * 100
    mean_squared_error = np.mean(np.square(error)) * 100

    return mean_percentage_error, mean_abs_percentage_error, mean_squared_error


def predictions_to_csv():
    """
    uses prediction numpy files to call evaluate_models for each file found
    and generate a csv output of organized evaluations of the reggression and classification
    models.
    """

    p = Path(".") / "data" / "predictions"
    print("Looking for prediction files")
    prediction_files = []
    for i in p.glob('*.npy'):
        if "prediction" in i.name:
            print(i.name)
            prediction_files.append(i.name)

    test_labels = np.load("data/ising/test_labels.npy")

    for i in prediction_files:
        curr = np.load("data/predictions/" + i)
        if "fmodel" in i:
            model_result = "final"
        elif "bmodel" in i:
            model_result = "best"
        else:
            warn("Model should be final or best, none detected")

        if "clas" in i:
            evaluate_models(model_iteration=int(i.split("_")[1]),
                            model_type="classification",
                            predicted=curr,
                            ground_truth=convert_to_categories(test_labels, 10),
                            model_result=model_result)

        elif "reg" in i:
            evaluate_models(model_iteration=int(i.split("_")[1]),
                            model_type="regression",
                            predicted=curr,
                            ground_truth=test_labels,
                            model_result=model_result)


def write_model_csv():

    # First get all the files we will need
    data = []
    p = Path(".") / "data" / "logs" / "model_logs"
    for i in p.glob("*.txt"):
        data.append(str(i.absolute()))

    write_header = True
    if os.path.exists("model_details.csv") is True:
            write_header = False

    field_names = ["Date", "Iteration", "Model Type", "Activation", "Batch Size",
                  "Dropout", "Batchnorm", "Batchnorm Order", "Learning Rate", "Sparse Data"]

    with open('model_details.csv', 'a+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)

        if write_header:
            writer.writeheader()

        for model in data:
            log_lines = []
            with open(model) as file: 
                for line in file:
                    log_lines.append(line.split(": ")[-1].replace("\n", "").strip())
            writer.writerow({"Date": log_lines[0],
                             "Iteration": log_lines[1],
                             "Model Type": log_lines[2],
                             "Activation": log_lines[3],
                             "Batch Size": log_lines[4],
                             "Dropout": log_lines[5],
                             "Batchnorm": log_lines[6],
                             "Batchnorm Order": log_lines[7],
                             "Learning Rate": "%f" % eval(log_lines[9]),
                             "Sparse Data": log_lines[10]})


def download_ising_data():
    """
    Downloads training/testing numpy files from s3 bucket using donload_from_s3
    """

    p = Path(".") / "data" / "ising"

    files = ["ising_data.npz", "test_data.npz", "test_labels.npy"]
    paths = [str(p.absolute()) + "/" + i for i in files]

    for file in paths:
        download_from_s3(file)


if __name__ == '__main__':

    if argv[1] == "download":
        download_last_files(download_models=False)
    elif argv[1] == "download_all":
        download_all_files(download_models=True)
    elif argv[1] == "upload":
        backup_files()
    elif argv[1] == "predictions":
        predictions_to_csv()
    elif argv[1] == "download_ising_files":
        download_ising_data()
    elif argv[1] == "write_model_details":
        write_model_csv()
    else:
        raise ValueError("Arg needs to be download, upload, predictions or download_ising_files")
