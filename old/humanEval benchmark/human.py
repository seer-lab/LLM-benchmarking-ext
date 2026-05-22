import openai
import anthropic
import requests
import json
import sys
import os
import re
import ast
import tempfile
import glob
from typing import Dict, List, Any, Tuple

# Replace with your API keys
openai_api_key = 
anthropic_api_key = 

def load_problems(directory: str = 'human_eval', problem_num: int = None) -> Dict[int, Any]:
    """
    Load JSON problem files from the human_eval directory.
    If problem_num is specified, load only that problem.
    Returns a dictionary with problem numbers as keys and problem data as values.
    """
    problems = {}
    
    # Check if directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found")
    
    # Determine which files to load
    if problem_num is not None:
        filenames = [f'problem{problem_num}.json']
    else:
        filenames = [f for f in os.listdir(directory) if f.startswith('problem') and f.endswith('.json')]
    
    for filename in filenames:
        try:
            # Extract problem number from filename
            prob_num = int(filename.replace('problem', '').replace('.json', ''))
            filepath = os.path.join(directory, filename)
            
            # Read and parse JSON file directly
            with open(filepath, 'r') as f:
                problem_data = json.load(f)
            
            problems[prob_num] = problem_data
            print(f"Successfully loaded {filepath}")
                
        except ValueError as e:
            print(f"Error parsing problem number from filename {filename}: {e}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {filename}: {e}")
        except Exception as e:
            print(f"Error loading file {filename}: {e}")
    
    if not problems:
        print(f"No valid problem files found in {directory} directory")
        return {}
        
    print(f"Successfully loaded {len(problems)} problem file(s)")
    return problems

def main(problems_dir: str = 'human_eval', problem_num: int = None):
    """
    Main function to test problems loaded from files.
    Args:
        problems_dir: Directory containing the problem files
        problem_num: Optional specific problem number to test (tests all if None)
    """
    global output_file
    output_filename = 'generated_outputs_problem{}.txt'.format(problem_num)
    with open(output_filename, 'w') as output_file:
        try:
            problems = load_problems(problems_dir, problem_num)
        except FileNotFoundError as e:
            print(e)
            return
        
        if not problems:
            return
        
        # List of model providers and their corresponding models
        model_providers = [
            ('openai', 'gpt-4'),                     # ChatGPT-4
            ('openai', 'gpt-3.5-turbo'),              # GPT-3.5-turbo
            ('anthropic', 'claude-3-5-sonnet-20241022'),        # Claude 3.5 Sonnet
            ('ollama', 'llama3.1:70b')                     # LLaMA 3.1 via Ollama
        ]
        
        for pid, problem_data in sorted(problems.items()):
            print_and_log(f"\n=== Testing Problem {pid} ===")
            print_and_log(f"Prompt: {problem_data.get('prompt', 'No prompt provided.')}")
            print_and_log("=" * 80)
            
            for provider, model in model_providers:
                print_and_log(f"\n--- Testing with {provider} ({model}) ---")
                passed = test_version(problem_data, provider, model)


def extract_python_code(text: str) -> str:
    """Extract only the Python code from the model's response."""
    # Try to find code between triple backticks with python specified
    code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Try to find code between triple backticks without python specified
    code_blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # If no code blocks found, attempt to extract code heuristically
    # Remove any leading explanations or texts
    code = re.sub(r'^[\s\S]*?(def\s)', r'\1', text, flags=re.DOTALL)
    return code.strip()

def get_code_from_llm_openai(prompt: str, model: str) -> Tuple[str, str]:
    """Generate code using OpenAI's models."""
    client = openai.OpenAI(api_key=openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": f"{prompt}\n\nPlease provide Python code for the following problem."}
            ],
            temperature=0
        )
        generated_response = response.choices[0].message.content
        code = extract_python_code(generated_response)
        return generated_response, code
    except Exception as e:
        print_and_log(f"Error with OpenAI model '{model}': {e}")
        return "", ""

def get_code_from_llm_anthropic(prompt: str, model: str) -> Tuple[str, str]:
    """Generate code using Anthropic's Claude models."""
    client = anthropic.Anthropic(api_key=anthropic_api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": f"{prompt}\n\nPlease provide Python code for the following problem."}
            ]
        )
        # Assuming response.content is a list of messages
        if isinstance(response.content, list) and len(response.content) > 0:
            generated_response = response.content[0].text.strip()
        else:
            generated_response = response.text.strip()
        code = extract_python_code(generated_response)
        return generated_response, code
    except Exception as e:
        print_and_log(f"Error with Anthropic model '{model}': {e}")
        return "", ""

def get_code_from_llm_ollama(prompt: str, model: str) -> Tuple[str, str]:
    """Generate code using Ollama's LLaMA models."""
    url = 'http://localhost:11434/api/generate'
    payload = {
        "model": model,
        "prompt": f"{prompt}\n\nPlease provide Python code for the following problem.",
        "temperature": 0
    }
    try:
        response = requests.post(url, json=payload, stream=True)
        if response.status_code == 200:
            generated_response = ''
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    generated_response += data.get('response', '')
            code = extract_python_code(generated_response)
            return generated_response, code
        else:
            print_and_log(f"Error from Ollama: {response.status_code} {response.text}")
            return "", ""
    except Exception as e:
        print_and_log(f"Error connecting to Ollama model '{model}': {e}")
        return "", ""

def get_code_from_llm(prompt: str, model_provider: str, model_name: str) -> Tuple[str, str]:
    """Dispatcher to get code from the specified LLM."""
    if model_provider == 'openai':
        return get_code_from_llm_openai(prompt, model_name)
    elif model_provider == 'anthropic':
        return get_code_from_llm_anthropic(prompt, model_name)
    elif model_provider == 'ollama':
        return get_code_from_llm_ollama(prompt, model_name)
    else:
        print_and_log(f"Unsupported model provider: {model_provider}")
        return "", ""

def find_main_function(code: str) -> str:
    """Extract the name of the main function from the generated code."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return node.name
    except Exception as e:
        print_and_log(f"Error parsing code: {e}")
    return None

def run_code_and_test(code: str, assertions: List[Dict[str, Any]]) -> bool:
    """Run the generated code and test it against assertions."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.py') as tmp_file:
        tmp_file_name = tmp_file.name
        tmp_file.write(code.encode('utf-8'))

    sys.path.insert(0, os.path.dirname(tmp_file_name))

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("temp_module", tmp_file_name)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)

        main_function_name = find_main_function(code)
        if not main_function_name:
            raise ValueError("Could not find a function definition in the generated code")
        
        main_function = getattr(temp_module, main_function_name)
        
    except Exception as e:
        print_and_log(f"Error importing generated code: {e}")
        os.unlink(tmp_file_name)
        sys.path.pop(0)
        return False

    all_passed = True
    for assertion in assertions:
        try:
            # Get all input parameters from the assertion
            input1 = assertion['input']
            expected_output = assertion['output']
            
            # Call the function with all input parameters unpacked
            result = main_function(**input1)
            
            # Handle tuple outputs by converting lists to tuples if necessary
            if isinstance(expected_output, list) and isinstance(result, tuple):
                expected_output = tuple(expected_output)
            elif isinstance(expected_output, list) and len(expected_output) == 2 and isinstance(result, tuple):
                # Special case for two-element lists that should be tuples
                expected_output = tuple(expected_output)
            
            # For floating point comparisons, use a small epsilon
            if isinstance(result, float) or (isinstance(result, tuple) and any(isinstance(x, float) for x in result)):
                is_equal = all(abs(r - e) < 1e-6 if isinstance(r, float) else r == e 
                             for r, e in zip(result, expected_output)) if isinstance(result, tuple) \
                          else abs(result - expected_output) < 1e-6
            else:
                is_equal = result == expected_output

            if is_equal:
                print_and_log(f"✓ Assertion passed for input {input1}")
            else:
                print_and_log(f"✗ Assertion failed for input {input1}")
                print_and_log(f"  Expected: {expected_output}")
                print_and_log(f"  Got:      {result}")
                all_passed = False
                
        except Exception as e:
            print_and_log(f"✗ Error during assertion for input {input1}: {e}")
            all_passed = False

    os.unlink(tmp_file_name)
    sys.path.pop(0)
    return all_passed


def test_version(problem_data: Dict[str, Any], model_provider: str, model_name: str) -> bool:
    """Test a single version of the problem using the specified model."""
    prompt = problem_data.get('prompt', 'No prompt provided.')
    assertions = problem_data.get('assertions', [])
    
    generated_response, code = get_code_from_llm(prompt, model_provider, model_name)
    if not code:
        print_and_log("✗ No code was generated.")
        return False
        
    print_and_log(f"**Generated Response:**\n{generated_response}\n")
    print_and_log(f"**Extracted Code:**\n{code}\n")
    
    passed = run_code_and_test(code, assertions)
    if passed:
        print_and_log(f"✓ All assertions passed")
    else:
        print_and_log(f"✗ Some assertions failed")
    
    return passed


def print_and_log(message: str):
    """Print a message to the console and log it to the output file."""
    print(message)
    if 'output_file' in globals() and output_file:
        output_file.write(message + '\n')

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test coding problems using various LLM providers')
    parser.add_argument('--problems-dir', default='human_eval',
                        help='Directory containing problem files')
    parser.add_argument('--problem', type=int, help='Specific problem number to test')
    
    args = parser.parse_args()
    main(problems_dir=args.problems_dir, problem_num=args.problem)
