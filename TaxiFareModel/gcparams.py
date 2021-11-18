### GCP Storage - - - - - - - - - - - - - - - - - - - - - -
BUCKET_NAME = 'wagon-data-737-hernandez'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -
# train data file location
BUCKET_TRAIN_DATA_PATH = 'data/train_1k.csv'
BUCKET_TEST_DATA_PATH = 'data/test.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -
# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'taxifare'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'v1'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME='wagon-data-737-hernandez'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# will store the packages uploaded to GCP for the training
BUCKET_TRAINING_FOLDER = 'trainings'

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# not required here

### Storage Location - - - - - - - - - - - - - - - - - - - -

STORAGE_LOCATION = 'models/simpletaxifare/model.joblib'
