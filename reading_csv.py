# Write a function that read csv file and return a list of lists
def read_csv(path):
    import csv
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    return data

# Write a function that read csv file a 2d array

def read_csv_array(path):
    import csv
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    return np.array(data)