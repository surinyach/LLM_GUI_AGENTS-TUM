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

SUBTASK_INSTRUCTIONS_CONTENT_TEMPLATE = "This is the task: {subtask} and these are the instructions: {instruction_list}"


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

    
    
    def set_subtask_and_instructions(self, subtask: str, instruction_list: str) -> None:
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
                instruction_list=instruction_list
            )
            
            content_to_add = genai.protos.Content(
                role='user',
                parts=[genai.protos.Part(text=content_message_string)]
            )

            self.chat.history.append(content_to_add)
            
            logger.info(f"Subtask '{subtask}' and its instructions added to chat history.")
        
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
    subtask = "dime lo que dice la primera pagina web sobre perros"
    instruction_list = """
    1. abre el navegador
    2. busca perros en la barra de búsqueda
    3. pulsa en el primer link
    4. copia el contenido del link que acabas de abrir
    """
    reflection_expert.set_subtask_and_instructions(subtask, instruction_list)

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
    1. pulsa la sección de imagenes
    2. haz click derecho sobre la imagen de un perro
    3. dale a guardar como
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
    1. pulsa la sección de imagenes
    2. haz click derecho sobre la imagen de un perro
    3. dale a guardar como
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



    
