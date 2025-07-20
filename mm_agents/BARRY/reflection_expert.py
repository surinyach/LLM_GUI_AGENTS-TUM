import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image


logger = logging.getLogger("reflection_expert")

EXECUTION_ERROR_REFLECTION_PROMPT_TEMPLATE = """
I had a problem executing the task, this is what i did: 

{action_expert_feedback} 

and this is why i could't do it:

{execution_error}

Give me feedback on how to solve the errors.
I also pass you a screenshot 
"""

FINISHED_INSTRUCTIONS_EVAL_PROMPT_TEMPLATE = """
Decide if the subtask is finished according to this feedback 

{action_expert_feedback} 

and the screenshot. Start the response with a 'Yes' or a 'No'. If it is no, explain why.
"""

SUBTASK_INSTRUCTIONS_CONTENT_TEMPLATE = "This is the subtask: '{subtask}'"

EVALUATE_EXECUTION_PROMPT = """
I have executed this instruction: '{instruction}'.
By what you see in my screen, do I have completed well?
Here's how I want you to structure your response:
1.  **Reasoning Process:** First, think step by step your thoughts. 
2.  **Revised instruction:** After your reasoning, you MUST respond only respond 'yes' or 'no' to indicate if the instruction was completed successfully.
The response MUST start with the exact phrase "RESPONSE:" on its own line, followed immediately by the response.

Example of how the response should appear:
RESPONSE: yes
"""

EVALUATE_ERROR_PROMPT = """
Tell me if the last error is a major or minor error. This is how you can classify an error:

Minor: The instruction can still be completed in the current state of the screen. For example:
 - the action agent has clicked the wrong button but with the current screen it is still possible to click the right button
 - It has been typed the wrong word but you can still deleted and write the correct one
 - The action expert couldn't click the right coordintates to move the slider but you can help him to aim better given the current position of the cursor.

Major: The instruction can not be completed anymore with the current state of the screen. For example:
 - The action expert clicked a button and the wrong page appeared.
 - The action expert tried to type on a search bar but the screen does not have a search bar
 - The action expert tried to click an icon that does not exist

You only have 3 chances in a row to solve the same minor error. If it keeps failing it is now a major error.

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
        Inicializa el experto en reflection.
        """
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY no encontrada en el archivo .env para ActionExpert")
            raise ValueError("GEMINI_API_KEY no encontrada en el archivo .env")
        
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_id)

        self.chat = self.model.start_chat(history=[])
        self.last_printed_index = 0

        self.instruction_list = []
        self.instruction_index = 0
        self.subtask = ""

        self.last_printed_index = 0 # this is for printing the chat history for debugging

    
    

    def set_subtask_and_instructions(self, subtask: str, instruction_list) -> None:
        """
        Adds the current subtask and its generated instructions to the chat history.

        This function serves to log the detailed plan (subtask and its specific
        instructions) within the ongoing conversation history. This ensures that
        the LLM (and anyone reviewing the chat history) has a clear record of
        the plan that was put into action.

        Args:
            subtask (str): The specific subtask that has been set for execution.
            instruction_list (str): The detailed list of instructions generated
                                    for carrying out the given subtask.

        Returns:
            None: This function directly modifies the `chat.history` in place.
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
            
            logger.info(f"Subtask '{subtask}' and first instruction added to chat history.")
        
        except Exception as e:
            logger.error(f"Error in set_subtask_and_instructions() of reflection_expert: {e}")
            raise

    
    def execution_error_reflection(self, execution_error: str, action_expert_feedback: str, SOM: any) -> str:
        """
        Facilitates reflection on an execution error by querying an LLM for feedback.

        This function provides the LLM with details of a failed execution, including
        the actions attempted and the specific error encountered. It asks the LLM
        for guidance on resolving the error.
        The current screen state (SOM) is also provided as context.

        Args:
            execution_error (str): A description of the error that occurred during task execution.
            action_expert_feedback (str): A summary of the actions that were attempted
                                          before the error occurred.
            SOM (any): The screenshot of the current screen state, providing visual context to the LLM.

        Returns:
            str: The LLM's textual response containing feedback on the error
        """
        prompt = EXECUTION_ERROR_REFLECTION_PROMPT_TEMPLATE.format(
            action_expert_feedback=action_expert_feedback,
            execution_error=execution_error
        )
        
        logger.info(f"Sending execution error details to LLM for reflection. Error: '{execution_error}'")

        try:
            response = self.chat.send_message([prompt, SOM])

            logger.info(f"LLM's error reflection feedback: {response.text}")
            return response.text

        except Exception as e:
            logger.error(f"Error in execution_error_reflection() of planning_expert: {e}")
            raise

    def finished_instructions_eval(self, action_expert_feedback: str, SOM: any) -> bool:
        """
        Evaluates whether a given subtask has been successfully completed.

        This function queries an LLM, providing it with feedback on actions performed
        by the action expert and the current screen state (SOM). The LLM's role
        is to determine if these actions, in conjunction with the visual context,
        indicate the successful completion of the current subtask. It expects a
        binary 'yes' or 'no' response from the LLM.

        Args:
            action_expert_feedback (str): A summary of the actions taken by the
                                          action expert for the current subtask.
            SOM (any): The screenshot of the current screen state, offering visual context to the LLM.

        Returns:
            bool: True if the LLM determines the subtask is finished ('yes'),
                  False otherwise ('no').

        Raises:
            Exception: If an error occurs during the interaction with the LLM.
        """
        try:
            prompt = FINISHED_INSTRUCTIONS_EVAL_PROMPT_TEMPLATE.format(
                action_expert_feedback=action_expert_feedback
            )
            
            logger.info(f"Asking LLM to evaluate subtask completion based on feedback: '{action_expert_feedback}'")

            response = self.chat.send_message([prompt, SOM])
            
            llm_decision = response.text.strip()
            logger.info(f"LLM's decision on subtask completion: '{llm_decision}'")

            return llm_decision

        except Exception as e:
            logger.error(f"Error in finished_instructions_eval() of reflection_expert: {e}")
            raise 
    
    def create_new_instruction(self):
        prompt = "taking into account the last evaluation respond only with the next instruction. don't add any comments."
        response =self.chat.send_message(prompt)
        return response.text

    def _print_history(self):
        for i in range(self.last_printed_index, len(self.chat.history)):
            message = self.chat.history[i]
            text_content = message.parts[0].text 
            print(f"{message.role}: {{ \"{text_content}\" }}")
        
        self.last_printed_index = len(self.chat.history)
    
    def evaluate_execution(self, screenshot):
        prompt = EVALUATE_EXECUTION_PROMPT.format(instruction = self.instruction_list[self.instruction_index])
        response =self.chat.send_message([prompt, screenshot])
        parts = response.text.split("RESPONSE:", 1)
        final_response = parts[1].strip()

        return final_response == "yes"
    
    def is_last_instruction(self):
        logger.info(f"estas son las intrucciones que tiene guardadas el reflection_expert {self.instruction_list} y esta el indice actual {self.instruction_index}")
        return len(self.instruction_list) - 1 == self.instruction_index
    
    def get_next_instruction(self):
        logger.info("doy la siguiente instrucción")
        self.instruction_index += 1
        return self.instruction_list[self.instruction_index]
    
    def evaluate_error(self, screenshot):
        prompt = EVALUATE_ERROR_PROMPT.format(instruction = self.instruction_list[self.instruction_index])
        response =self.chat.send_message([prompt, screenshot])
        parts = response.text.split("RESPONSE:", 1)
        final_response = parts[1].strip()

        return final_response

    def _print_history(self):
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

        




    
    
    """
    # caso 1 --------------------------------
    image_name = 'som_screenshot_4.png'
    image_path = os.path.join(image_folder, image_name)
    SOM = Image.open(image_path)

    osworld_action = "error: no he encontrado ningún link"
    action_expert_feedback = "he abierto el navegador, he buscado perros en la barra de busqueda"

    print("caso 1: execution error por parte del action expert")

    if osworld_action.startswith("error:"):
        reflection_expert_feedback = reflection_expert.execution_error_reflection(
            osworld_action,
            action_expert_feedback,
            SOM)
    reflection_expert._print_history()

    print("caso 1 terminado -----------------------------------------------\n")

    
    # caso 2.1 ---------------------------------
    image_name = 'som_screenshot_5.png'
    image_path = os.path.join(image_folder, image_name)
    SOM = Image.open(image_path)

    subtask = "descargate la foto de un perro"
    instruction_list = """
    #1. pulsa la sección de imagenes
    #2. haz click derecho sobre la imagen de un perro
    #3. dale a guardar como
    """
    reflection_expert.set_subtask_and_instructions(subtask, instruction_list)
    action_expert_feedback = "He pulsado en la sección de imagenes después he pulsado click derecho sobre la imagen de un perro y la he guardado "

    print("caso 2.1: Instruction list terminada pero incorrectamente")

    reflection_expert_feedback = reflection_expert.finished_instructions_eval(
        action_expert_feedback,
        SOM) 
    if reflection_expert_feedback.startswith("Yes"):
        reflection_expert_feedback = ""
        print("Ha terminado correctamente")
    else:
        print("NO ha terminado correctamente")
    
    reflection_expert._print_history()
    
    print("caso 2.1 terminado -----------------------------------------------\n")

    # caso 2.2 ---------------------------------
    image_name = 'som_screenshot_6.png'
    image_path = os.path.join(image_folder, image_name)
    SOM = Image.open(image_path)

    subtask = "descargate la foto de un perro"
    instruction_list = """
    #1. pulsa la sección de imagenes
    #2. haz click derecho sobre la imagen de un perro
    #3. dale a guardar como
    """
    reflection_expert.set_subtask_and_instructions(subtask, instruction_list)
    action_expert_feedback = "He pulsado en la sección de imagenes después he pulsado click derecho sobre la imagen de un perro y la he guardado "

    print("caso 2.2: Instruction list terminada correctamente")

    reflection_expert_feedback = reflection_expert.finished_instructions_eval(
        action_expert_feedback,
        SOM) 
    if reflection_expert_feedback.startswith("Yes"):
        reflection_expert_feedback = ""
        print("Ha terminado correctamente")
    else:
        print("NO ha terminado correctamente")
    
    reflection_expert._print_history()
    
    print("caso 2.2 terminado -----------------------------------------------\n")
    """


    
