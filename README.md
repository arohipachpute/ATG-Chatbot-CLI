# ATG Technical Assignment: Modular CLI Chatbot (microsoft/DialoGPT-small)

This project develops a fully functional **Local Command-Line Interface (CLI) Chatbot** using the Hugging Face `transformers` library. It adheres to all requirements, including utilizing a sliding window mechanism for conversational context and implementing a robust, modular code structure.

## Modular Code Structure

The solution is organized into three distinct, required modules:

* **`model_loader.py`**: Handles loading the `microsoft/DialoGPT-small` model, tokenizer, and initializing the Hugging Face `pipeline` on the CPU device.
* **`chat_memory.py`**: Implements the sliding window memory logic using a Python `deque` to manage the context of the last **4 conversational turns**.
* **`interface.py`**: Contains the main CLI loop, user input handling, and integrates a custom `FACTS_DB` lookup layer for guaranteed accuracy on simple factual questions.

---

## Setup and Local Execution Instructions

The chatbot is designed to run locally in a command-line environment (Windows CMD/PowerShell or Linux/macOS terminal).

### **1. Dependency Installation**

1.  **Dependencies:** This project requires `torch` and `transformers`.
2.  **Install:** Run the following commands inside your activated virtual environment (recommended):
    ```bash
    # Install the core ML libraries (CPU-only for stability)
    pip install transformers
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
    ```

3.  **Note on Execution:** Initial local execution encountered a `WinError 1114` (DLL initialization failure) due to a missing Microsoft Visual C++ Redistributable. This was successfully resolved by repairing the necessary system files, allowing the project to run **locally** as intended.

### **2. Running the Chatbot**

1.  Ensure all three Python files are in the same directory.
2.  Run the main interface script from your terminal:
    ```bash
    python interface.py
    ```

---

## Sample Interaction Examples

The chatbot accepts continuous input, maintains context, and terminates gracefully on `/exit`.
