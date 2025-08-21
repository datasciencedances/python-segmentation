import yaml

def load_config_yaml(file_path):
    """
    Đọc file YAML và trả về nội dung dưới dạng dict.

    Parameters:
        file_path (str): Đường dẫn đến file .yaml hoặc .yml

    Returns:
        dict: Nội dung file YAML dưới dạng dictionary
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config