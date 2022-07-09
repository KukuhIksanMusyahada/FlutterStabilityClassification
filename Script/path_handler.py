import os

def get_this_dir():
    return os.path.dirname( os.path.abspath(__file__) )

def get_data_source():
    return os.path.join(get_this_dir(), os.pardir, 'Data_Source')

def get_raw_data():
    return os.path.join(get_data_source(), 'Raw')

def get_processed_data():
    return os.path.join(get_data_source(), 'Processed_Data')

def get_models_data():
    return os.path.join(get_data_source(), 'Models_Data')



