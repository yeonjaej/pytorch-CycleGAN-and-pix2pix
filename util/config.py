import yaml

def load_train_image_filename(yaml_filename):
    filename = '?'
    with open(yaml_filename) as f:
        yaml_file = yaml.safe_load(f);
        filename = yaml_file['train-image-filename']
    return filename
    
def load_valid_image_filename(yaml_filename):
    filename = '?'
    with open(yaml_filename) as f:
        yaml_file = yaml.safe_load(f);
        filename = yaml_file['valid-image-filename']
    return filename

def load_test_image_filename(yaml_filename):
    filename = '?'
    with open(yaml_filename) as f:
        yaml_file = yaml.safe_load(f);
        filename = yaml_file['test-image-filename']
    return filename

def load_train_label_filename(yaml_filename):
    filename = '?'
    with open(yaml_filename) as f:
        yaml_file = yaml.safe_load(f);
        filename = yaml_file['train-label-filename']
    return filename

def load_valid_label_filename(yaml_filename):
    filename = '?'
    with open(yaml_filename) as f:
        yaml_file = yaml.safe_load(f);
        filename = yaml_file['valid-label-filename']
    return filename

def load_test_label_filename(yaml_filename):
    filename = '?'
    with open(yaml_filename) as f:
        yaml_file = yaml.safe_load(f);
        filename = yaml_file['test-label-filename']
    return filename

def load_train_image_filenames(yaml_filenames):
    filenames = '?'
    with open(yaml_filenames) as f:
        yaml_file = yaml.safe_load(f);
        filenames = yaml_file['train-image-filenames']
    return filenames

def load_valid_image_filenames(yaml_filenames):
    filenames = '?'
    with open(yaml_filenames) as f:
        yaml_file = yaml.safe_load(f);
        filenames = yaml_file['valid-image-filenames']
    return filenames

def load_test_image_filenames(yaml_filenames):
    filenames = '?'
    with open(yaml_filenames) as f:
        yaml_file = yaml.safe_load(f);
        filenames = yaml_file['test-image-filenames']
    return filenames

def load_train_label_filenames(yaml_filenames):
    filenames = '?'
    with open(yaml_filenames) as f:
        yaml_file = yaml.safe_load(f);
        filenames = yaml_file['train-label-filenames']
    return filenames

def load_valid_label_filenames(yaml_filenames):
    filenames = '?'
    with open(yaml_filenames) as f:
        yaml_file = yaml.safe_load(f);
        filenames = yaml_file['valid-label-filenames']
    return filenames

def load_test_label_filenames(yaml_filenames):
    filenames = '?'
    with open(yaml_filenames) as f:
        yaml_file = yaml.safe_load(f);
        filenames = yaml_file['test-label-filenames']
    return filenames
