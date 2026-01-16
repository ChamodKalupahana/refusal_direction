"""
Convert jailbreakbench completions to chat format.
"""
import json
import os

def convert_to_chat_format(input_file, output_file):
    """Convert completions JSON to chat format."""
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    converted = []
    for item in data:
        prompt = item['prompt']
        response = item['response']
        
        # Create the chat format prompt
        chat_prompt = f"""<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{response}
"""
        converted.append({"prompt": chat_prompt})
    
    with open(output_file, 'w') as f:
        json.dump(converted, f, indent=2)
    
    print(f"Converted {len(converted)} items from {input_file} to {output_file}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Convert baseline completions
    convert_to_chat_format(
        os.path.join(script_dir, "llama3_jailbreakbench_baseline_completions.json"),
        os.path.join(script_dir, "llama3_jailbreakbench_baseline_chat_format.json")
    )
    
    # Convert actadd completions
    convert_to_chat_format(
        os.path.join(script_dir, "llama3_jailbreakbench_actadd_completions.json"),
        os.path.join(script_dir, "llama3_jailbreakbench_actadd_chat_format.json")
    )


if __name__ == "__main__":
    main()
