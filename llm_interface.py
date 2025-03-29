"""
Module to interact with various Large Language Models (LLMs).

Requires API keys and necessary libraries (e.g., openai, anthropic).
Store API keys securely (e.g., environment variables), do not hardcode them.
"""
import os
import re
try:
    import google.generativeai as genai
except ImportError:
    genai = None # Handle case where library is not installed

def generate_code(llm_name: str, prompt: str) -> str | None:
    """
    Sends a prompt to the specified LLM and returns the generated code.

    Args:
        llm_name: The identifier for the LLM (e.g., 'gpt-4', 'claude-3-opus').
        prompt: The prompt to send to the LLM.

    Returns:
        The generated code as a string, or None if an error occurs.
    """
    print(f"Placeholder: Generating code from {llm_name} with prompt: '{prompt[:50]}...'")
    # Replace with actual API calls to different LLMs
    if llm_name == "Gemini 2.5 Pro Exp":
        if genai is None:
            print("Error: google.generativeai library not installed. Run 'pip install google-generativeai'")
            return None
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                print("Error: GEMINI_API_KEY environment variable not set.")
                return None

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')

            # Optional: Add safety settings if desired
            # safety_settings = [
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            # ]
            # response = model.generate_content(prompt, safety_settings=safety_settings)

            response = model.generate_content(prompt)

            # Basic check for response content
            if not response.parts:
                 print(f"Error: Gemini API returned no parts in the response. Response: {response}")
                 return None

            # Extract text content - handle potential lack of text
            generated_text = ""
            try:
                 generated_text = response.text
            except ValueError:
                 # Handle cases where response.text might raise ValueError (e.g., blocked content)
                 print(f"Error: Could not extract text from Gemini response. It might be blocked. Full response: {response}")
                 return None # Or handle based on response.prompt_feedback

            # Extract Python code block (handles ```python ... ``` or just ```...```)
            # Regex to find code blocks fenced by triple backticks, optionally with 'python' tag
            code_match = re.search(r"```(?:python\n)?(.*?)```", generated_text, re.DOTALL | re.IGNORECASE)
            if code_match:
                return code_match.group(1).strip()
            else:
                # If no fenced block, assume the whole response might be code (less reliable)
                print("Warning: Could not find Python code block in ``` markers. Returning entire response.")
                return generated_text.strip()

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return None
    # Add placeholders for other LLMs here if needed
    # elif llm_name == "gpt-4":
    #     # Call OpenAI API
    #     pass
    # elif llm_name == "claude-3-opus":
    #     # Call Anthropic API
    #     pass
    else:
        print(f"LLM '{llm_name}' not implemented.")
        return None

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Example for Gemini (requires GOOGLE_API_KEY env var and library)
    # test_prompt = "Generate a Python function for bubble sort."
    # generated_code = generate_code("Gemini 2.5 Pro Exp", test_prompt)
    # if generated_code:
    #     print("\nGenerated Code:\n", generated_code)
    # else:
    #     print("Failed to generate code.")
    print("llm_interface.py executed directly. No default action.")
