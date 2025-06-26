import os
import json

def process_json_files(directory):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "true_labels.txt")
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        for filename in os.listdir(directory):
            print('Found: ', filename)
            if filename.endswith(".json"):
                file_path = os.path.join(directory, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as infile:
                        data = json.load(infile)
                        region = data.get("location__region", "N/A")
                        sample_name = os.path.splitext(filename)[0] 
                        line = f"{sample_name} : {region}\n"
                        outfile.write(line)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

def main():
    directory = "/home/mene/Downloads/json wget/"
    if os.path.isdir(directory):
        process_json_files(directory)
        print("Data extracted and saved to true_labels.txt")
    else:
        print("Invalid directory. Please try again.")

if __name__ == "__main__":
    main()