import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from .utils import parse_llm_response


logger = logging.getLogger("reflection_expert")

SUBTASK_INSTRUCTIONS_CONTENT_TEMPLATE = "This is the subtask: '{subtask}'"


FIRST_EVALUATE_EXECUTION_PROMPT = """
You are an expert on evaluating execution of actions in GUI environments.

I have executed this instruction: {instruction}

Task:

The attached image shows the state after executing the instruction. 
Reason which elements are important to analyze in order to determine if the instruction indicated before has been successfully solved.
Reflect carefully on what the *intended outcome* of the instruction would look like on the screen *after* execution.

Response:

Description of the critical elements that need to be inspected to decide if the instruction has been acomplished. 
Also add in the description in which is the expected state of every of these elements after the instruction has been correctly executed in the environment. 

Example:
If the instruction is to remove something, the current screen must not show the item to be a successful execution.
"""

SECOND_EVALUATE_EXECUTION_PROMPT="""
You are an image analyzer expert.

Task:

Among all the descripted elements of the previous analysis, choose which are the most important ones. 
Then analyze them in detail in the screen to see if it accomplishes the desired state.

Response:

Give the detailed description of the state of the most important elements.
Then provide also a reasoning about if the state of the elements is the desired one after completing the instruction in the current screen of the environment showed in the screenshot. 
"""

THIRD_EVALUATE_EXECUTION_PROMPT="""
Task:

Taking into account the previous analysis, the expected state of the elements after executing the instruction and the state of the screen determine if the instruction has been correctly performed or not.

Crucial point: The showed screen is the state after executing the instruction. Therefore, you must evaluate the success by observing the changes or absence of elements that the instruction aimed to alter or remove.
You should trust the systems more than you trust you so if the system says something is visible it is true. Take into account that you are not good reconizing things.
IMPORTANT: If you have done an action that must work like pressing ctrl + shift + b, the execution will be always successful, you just have to take into account that if there is no text you won't be able to see it but you have to act s if it was there.

For example:

* If the instruction was to "close the pop-up," and the pop-up is no longer visible on the screen, that indicates successful completion, not an error.
* If the instruction was to "add a new folder," and a new folder is now visible, that indicates successful completion.
* If the instruction was to "delete an item," and that item is no longer on the screen, that indicates successful completion.

Response:

You MUST respond only 'yes' or 'no' to indicate if the instruction was completed successfully based on the screen.
The response MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the response.

Example of how the response should appear:
RESPONSE: yes
"""

EVALUATE_ERROR_PROMPT = """
Tell me if the last error is a major or minor error. 
Also take into account that when searching in a browser the first links usually are patrocinated and we are not intereseted in them.
Think about what you are looking for and see if the first link makes sense.
Think also if this step is necessary for the main task {main_task}
This is how you can classify an error:

Minor: The instruction can still be completed in the current state of the screen with ONLY ONE INSTRUCTION, if more than one instruction is need it then It is a MAJOR!! For example:
 - the action agent has clicked the wrong button but with the current screen it is still possible to click the right button (only one instruction)
 - It has been typed the wrong word but you can still deleted and write the correct one (it is only one instruction)
 - The action expert couldn't click the right coordintates to move the slider but you can help him to aim better given the current position of the cursor.
 - The action expert had to click a link but it is needed to scroll down to be able to see the link

Major: The instruction can not be completed anymore with the current state of the screen. For example:
 - The action expert clicked a button and the wrong page appeared.
 - The action expert tried to type on a search bar but the screen does not have a search bar
 - The action expert tried to click an icon that does not exist

You only have 2 chances in a row to solve the same minor error. If it keeps failing it is now a major error.
Think if scrolling is needed to find what you are looking for

Here's how I want you to structure your response:
1.  **Reasoning Process:** First, think step by step your thoughts. 
2.  **Revised instruction:** After your reasoning, you MUST respond with what caused the error and different solutions to solve it.
The response MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the response.

Example of how the response should appear:
RESPONSE: Minor: You should click slightly more to the right.
"""


class ReflectionExpert:
    def __init__(self, model_id: str = "gemini-2.0-flash"):
        """
        Initialitation of the Reflexion Expert
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY not found in the file .env for Reflection Expert")
            raise ValueError("GEMINI_API_KEY not found in the file .env")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_id)

        self.chat = self.model.start_chat(history=[])

        self.instruction_list = []
        self.instruction_index = 0
        
        self.last_printed_index = 0 # this is for printing the chat history for debugging
    
    

    def set_subtask_and_instructions(self, subtask: str, instruction_list) -> None:
        """
        Saves the instruction_list and adds the current subtask to the chat history.

        This function serves to log the detailed plan (subtask and its specific
        instructions) within the ongoing conversation history. This ensures that
        the LLM (and anyone reviewing the chat history) has a clear record of
        the plan that was put into action. It also initializes the instruction
        index to zero, preparing for the sequential execution of the instructions.

        Args:
            subtask (str): The specific subtask that has been set for execution.
            instruction_list (list): The detailed list of instructions generated
                                     for carrying out the given subtask. This is
                                     expected to be a list of strings, where each
                                     string is a distinct instruction.

        Returns:
            None: This function directly modifies the `chat.history` by appending
                  a user message, and it updates `self.instruction_list` and
                  `self.instruction_index`.
        """
        try:
            content_message_string = SUBTASK_INSTRUCTIONS_CONTENT_TEMPLATE.format(
                subtask=subtask,
            )
            
            content_to_add = genai.protos.Content(
                role='user',
                parts=[genai.protos.Part(text=content_message_string)]
            )

            self.chat.history.append(content_to_add)
            self.instruction_list = instruction_list
            self.instruction_index = 0
            
        
        except Exception as e:
            logger.error(f"Error in set_subtask_and_instructions() of reflection_expert: {e}")
            raise
    
    def create_new_instruction(self):
        """
        Generates a new, single instruction to address a minor error without altering the existing instruction list.

        This function is invoked when a small issue can be resolved with a one-off instruction.
        It prompts the language model (LLM) to generate a corrective instruction based on the
        last evaluation. Crucially, this new instruction is not added to the ongoing `instruction_list`,
        nor does it modify the sequence of existing instructions. The agent is expected to execute
        this generated instruction immediately and then resume its progression through the
        original `instruction_list` as if no error had occurred.

        Args:
            None as it has all the information in the chat history.

        Returns:
            str: The newly generated single instruction from the LLM.
        """
        prompt = "Taking into account the last evaluation, respond only with the next instruction. don't add any comments."
        response =self.chat.send_message(prompt)
        return response.text

    def evaluate_execution(self, screenshot):
        """
        Evaluates the success of an executed instruction by interacting with a language model
        (LLM) in a multi-step conversational process.


        Args:
            self: The instance of the class, expected to have a `chat` object for sending messages
                and an `instruction_list` with the `instruction_index` pointing to the current instruction.
            screenshot: The image representing the state of the GUI after the instruction was executed.

        Returns:
            bool: True if the LLM determines the instruction was successfully completed ('yes'),
                False otherwise ('no').
        """
        try:
            prompt = FIRST_EVALUATE_EXECUTION_PROMPT.format(instruction = self.instruction_list[self.instruction_index])
            response = self.chat.send_message([screenshot, prompt])
            response = self.chat.send_message(SECOND_EVALUATE_EXECUTION_PROMPT)
            response = self.chat.send_message(THIRD_EVALUATE_EXECUTION_PROMPT)

            final_response = parse_llm_response(response.text)

            logger.info("Did the execution went well? " + final_response)

            return final_response.lower() == 'yes'
        
        except Exception as e:
            logger.error(f"Error in evaluate_execution: {e}")
            raise
    
    def is_last_instruction(self) -> bool:
        """
        Checks if the current instruction being processed is the last one in the instruction list.

        This function is crucial for controlling the flow of instruction execution,
        allowing the system to determine when all planned steps have been completed.

        Args:
            self: The instance of the class, expected to have an `instruction_list`
                (a list of instructions) and an `instruction_index` (the index
                of the currently active instruction).

        Returns:
            bool: True if the `instruction_index` points to the last element of
                `instruction_list`, False otherwise.
        """
        return len(self.instruction_list) - 1 == self.instruction_index
    
    def get_next_instruction(self):
        """
        Retrieves the next instruction from the instruction list and advances the instruction index.
        Always the function is_last_instruction() should be called before to prevent index out of bound errors.

        This function increments the internal `instruction_index` and then returns the
        instruction at this new index. It is used to get the next instruction to pass it to the action expert.

        Args:
            self: The instance of the class, expected to have an `instruction_list`
                (a list of instructions) and an `instruction_index` (the index
                of the currently active instruction).

        Returns:
            str: The instruction string at the newly incremented `instruction_index`.
        """
        self.instruction_index += 1
        return self.instruction_list[self.instruction_index]
    
    def evaluate_error(self, main_task, screenshot):
        """
        Evaluates a detected error by querying a language model (LLM) to classify it
        as minor or major and suggest solutions.

        This function sends the current instruction, the main task, and a screenshot
        to the LLM. The LLM then provides a detailed response that includes a reasoning
        process, identifies the cause of the error, and proposes solutions. The error
        classification (minor or major) is based on specific criteria defined in the
        `EVALUATE_ERROR_PROMPT`, primarily whether the error can be resolved with a
        single follow-up instruction.

        Args:
            self: The instance of the class, providing access to the `chat` object
                for LLM interaction and the `instruction_list` with the current `instruction_index`.
            main_task (str): A description of the overall main task, used by the LLM
                            for context in evaluating the necessity of steps.
            screenshot: The image representing the current state of the GUI where the
                        error occurred.

        Returns:
            str: The LLM's response, stripped of the "RESPONSE:" prefix, which
                contains the error classification (e.g., "Minor: ...", "Major: ...")
                and proposed solutions.
        """
        try:

            prompt = EVALUATE_ERROR_PROMPT.format(instruction = self.instruction_list[self.instruction_index], main_task = main_task)
            response =self.chat.send_message([prompt, screenshot])

            logger.info("This is the response of the reflection expert: " + response.text)

            final_response = parse_llm_response(response.text)
            return final_response
        
        except Exception as e:
            logger.error(f"Error in evaluate_error: {e}")
            raise
    
    

    def _print_history(self):
        """
        Prints new messages from the chat history to the console since the last print.

        This internal helper function iterates through the `chat.history` list, starting
        from the `last_printed_index`. It prints each new message, formatted to show
        its role (e.g., 'user', 'model') and the text content. After printing, it updates
        `last_printed_index` to the current total number of messages in the history,
        ensuring that only new messages are printed in subsequent calls.

        Args:
            self: The instance of the class, expected to have a `chat` object with a
                `history` attribute (a list of message objects) and a `last_printed_index`
                attribute (an integer tracking the last printed message's index).

        Returns:
            None: This function performs side effects by printing to standard output
                and modifying `self.last_printed_index`.
        """
        for i in range(self.last_printed_index, len(self.chat.history)):
            message = self.chat.history[i]
            text_content = message.parts[0].text 
            print(f"{message.role}: {{ \"{text_content}\" }}")
        
        self.last_printed_index = len(self.chat.history)

    

if __name__ == "__main__":
    current_dir = os.path.dirname(__file__) # Obtiene el directorio del script actual
    image_folder = os.path.join(current_dir, 'mocks')

    reflection_expert = ReflectionExpert()
    subtask = "buscar imagenes de perros"
    instruction_list = [
    "abre el navegador",
    "pulsar en la sección de imagenes",
    "pulsa en el primer link",
    "copia el contenido del link que acabas de abrir"
    ]

    image_name = 'som_screenshot_2.png'
    image_path = os.path.join(image_folder, image_name)
    screenshot = Image.open(image_path)

    reflection_expert.set_subtask_and_instructions(subtask, instruction_list)
    successful = reflection_expert.evaluate_execution(screenshot)

    # case 1
    if successful:
        is_last_instruction = reflection_expert.is_last_instruction()
        if is_last_instruction:
            print("""return {
                'reflection_planning': 'finish',
                'reflection_action': ''
            }""")
        else:
            next_instruction = reflection_expert.get_next_instruction()
            #action_expert.set_current_instruction(next_instruction)
            print("""return {
                "reflection_planning": "",
                "reflection_action": ""
            }""")
    
    reflection_expert._print_history()

    
    # case 2
    image_name = 'som_screenshot_3.png'
    image_path = os.path.join(image_folder, image_name)
    screenshot = Image.open(image_path)

    successful = reflection_expert.evaluate_execution(screenshot)

    evaluated_error = reflection_expert.evaluate_error(screenshot)
    if evaluated_error.startswith("Minor:"):
        print("""return {
            "reflection_action": evaluated_error,
            "reflection_planning": ""
        }""")
    else:
        print("""return {
            "refelction_action": "",
            "reflection_planning": evaluated_error
        }""")

    reflection_expert._print_history()