import os
from llm_adapter import generate_solution

if __name__ == '__main__':
    print('LLM_PROVIDER =', os.getenv('LLM_PROVIDER', 'openai'))
    prompt = 'Write a short Python function that returns the sum of two integers.'
    try:
        out = generate_solution(prompt)
        print('\n=== OUTPUT START ===\n')
        print(out)
        print('\n=== OUTPUT END ===')
    except Exception as e:
        print('Error calling LLM adapter:', e)
