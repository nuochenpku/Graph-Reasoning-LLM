import argparse

def split_json_file(task, input_file_path):

    with open(input_file_path, 'r') as original_file,\
         open(f"../test_set_v2_9000/{task}_test.json", 'w') as odd_file,\
         open(f"../train_set_v3/{task}_train.json", 'w') as even_file:
        
        for index, line in enumerate(original_file):
            if index % 2 == 0:
                odd_file.write(line)
            else:
                even_file.write(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a JSON file into odd and even lines.")
    parser.add_argument('--task', type=str, default="cycle", help="The task name for the output file.")
    args = parser.parse_args()
    input_file = f"{args.task}_test.json"

    split_json_file(args.task, input_file)