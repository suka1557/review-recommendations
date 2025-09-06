import ijson
import json

def read_json_array(filepath, num_items=None):
    """
    Reads a specified number of items from a large JSON array file.
    
    Args:
        filepath (str): Path to the JSON file.
        num_items (int or None): Number of items to read. If None, reads the whole array.
        
    Returns:
        list: List of parsed JSON objects.
    """
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        parser = ijson.items(f, 'item')
        if num_items is None:
            items = list(parser)
        else:
            for i, obj in enumerate(parser):
                if i >= num_items:
                    break
                items.append(obj)
    return items

# Example usage:
# sample = read_json_array('/path/to/your/large_array.json', num_items=5)
# print(sample)


def read_json_lines(filepath, num_items=None):
    """
    Reads a specified number of items from a JSON Lines file.
    
    Args:
        filepath (str): Path to the JSON file.
        num_items (int or None): Number of items to read. If None, reads the whole file.
        
    Returns:
        list: List of parsed JSON objects.
    """
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        if num_items is None:
            for line in f:
                items.append(json.loads(line))
        else:
            for i, line in enumerate(f):
                if i >= num_items:
                    break
                items.append(json.loads(line))
    return items


def json_lines_to_dataframe(filepath, num_items=None):
    """
    Reads a JSON Lines file and returns a pandas DataFrame with all fields as columns.
    
    Args:
        filepath (str): Path to the JSON Lines file.
        num_items (int or None): Number of lines to read. If None, reads the whole file.
        
    Returns:
        pd.DataFrame: DataFrame with each field as a column.
    """
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_items is not None and i >= num_items:
                break
            record = json.loads(line)
            records.append(record)
    return records

def save_dict_to_json(data: dict, filepath: str):
    """
    Saves a dictionary to a JSON file with indentation of 4.
    
    Args:
        data (dict): The dictionary to save.
        filepath (str): The path to the output JSON file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)