import os
import glob

files = glob.glob('*_agent.py')
for f in files:
    if f == 'simulation_executor_agent.py':
        continue
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    if 'model_info={' not in content:
        content = content.replace(
            "model=model, \n            api_key=api_key, \n            base_url=base_url,",
            "model=model, \n            api_key=api_key, \n            base_url=base_url,\n            model_info={'vision': False, 'function_calling': True, 'json_output': True, 'family': 'unknown'},"
        )
        content = content.replace(
            "model=model,\n            api_key=api_key,\n            base_url=base_url,",
            "model=model,\n            api_key=api_key,\n            base_url=base_url,\n            model_info={'vision': False, 'function_calling': True, 'json_output': True, 'family': 'unknown'},"
        )
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Patched {f}")
