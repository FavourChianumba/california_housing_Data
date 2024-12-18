import pandas as pd

def load_data(file_name):
    """
    Load data from a specified CSV file in the data folder.
    
    :param file_name: Name of the CSV file to load (e.g., 'housing.csv' or 'California_Houses.csv').
    :return: A pandas DataFrame containing the data.
    """
    data_path = f'../data/{file_name}'
    
    try:
        data = pd.read_csv(data_path)
        print(f"Successfully loaded data from {file_name}.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_name}' does not exist in the '../data/' directory.")
    except Exception as e:
        raise ValueError(f"An error occurred while loading the file '{file_name}': {e}")

