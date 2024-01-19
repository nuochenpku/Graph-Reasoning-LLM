# import argparse

# def split_json_file(task, input_file_path):

#     with open(input_file_path, 'r') as original_file,\
#          open(f"../test_set_v2_9000/{task}_test.json", 'w') as odd_file,\
#          open(f"../train_set_v3/{task}_train.json", 'w') as even_file:
        
#         for index, line in enumerate(original_file):
#             if index % 2 == 0:
#                 odd_file.write(line)
#             else:
#                 even_file.write(line)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Split a JSON file into odd and even lines.")
#     parser.add_argument('--task', type=str, default="cycle", help="The task name for the output file.")
#     args = parser.parse_args()
#     input_file = f"{args.task}_test.json"

#     split_json_file(args.task, input_file)

import argparse

def split_json_file(task, input_file_path):

    with open(input_file_path, 'r') as original_file,\
         open(f"../test_set_v4_3600/{task}_test.json", 'w') as output_file:

        chunk_size = 200  # Define the chunk size
        sample_size = 80  # Define the sample size to take from each chunk
        lines_written = 0  # Initialize a counter for lines written to the output file
        
        for index, line in enumerate(original_file):
            # Calculate the index within the current chunk (0-199)
            chunk_index = index % chunk_size

            # Write the line if it is within the first 80 lines of the current chunk
            if chunk_index < sample_size:
                output_file.write(line)
                lines_written += 1

            # Stop processing after 5 chunks (1000 lines)
            if index >= chunk_size * 5 - 1:
                break

        print(f"Total lines written: {lines_written}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract specific lines from a JSON file.")
    parser.add_argument('--task', type=str, required=True, help="The task name for the output file.")
    
    # Add an argument for specifying the input file path
    parser.add_argument('--input', type=str, required=False, help="The input JSON file path.")
    args = parser.parse_args()
    input_file = f"../test_set_v3_9000/{args.task}_test.json"
    split_json_file(args.task, input_file)