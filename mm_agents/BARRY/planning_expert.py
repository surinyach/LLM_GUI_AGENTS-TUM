import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

logger = logging.getLogger("planning_expert")

class PlanningExpert:
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

        self.main_task = ""
        self.instruction_list = ""
        self.subtask_list = ""
        self.current_subtask = 0

        self.first_iter = True

    
    def parse_subtask_response(self, response_text: str) -> List[str]:
        """
        Parses the raw text response from the LLM into a list of subtasks.
        Assumes tasks are separated by ';'.
        """
        if not response_text:
            return []
        
        # Split by semicolon, strip whitespace, and filter out empty strings
        subtasks = [task.strip() for task in response_text.split(';') if task.strip()]
        return subtasks
    
    def save_main_task(self, main_task):
        logger.info("planning expert guarda y descompone main task")

        try:
            self.main_task = main_task
            prompt = f"This is the main task, decompose it into subtasks, separating each subtask by a semicolon ';': {main_task}"
            response = self.chat.send_message(prompt)
            self.subtask_list = self.parse_subtask_response(response.text)
            logger.info(f"estas son las subtareas que ha creado {response.text}")


            self.current_subtask = 0
            return self.subtask_list[0]
        
        except Exception as e:
            logger.error(f"Error en la función save_main_task() del planning_expert: {e}")
            raise
    
    def decide_if_task_finished(self, reflection_expert_feedback):
        try:
            prompt = f"Decide if the subtask: {self.subtask_list[self.current_subtask]} is finished according to this feedback {reflection_expert_feedback}. Respond only with a 'yes' or 'no' don't add any more comments"
            response = self.chat.send_message(prompt)
            logger.info(f"decisión de si la tarea ha acabado: {response.text}")

            return response.text.strip() == "yes"

        except Exception as e:
            logger.error(f"Error en la función decide_if_task_finished() del planning_expert: {e}")
            raise
    
    def rethink_instruction_list(self, action_expert_feedback, SOM):
        logger.info(" toca repensar la lista de instrucciones")
        try:
            prompt = f"""This is what i did: {action_expert_feedback} take also into account the feedback mentioned in the previous message. 
            Taking this into account and the state of my screen right now can you rethink this subtask?: {self.subtask_list[self.current_subtask]}"""
            response = self.chat.send_message([prompt, SOM])
            logger.info(f"esta es la nueva lista de instrucciones: {response.text}\n")

            return response.text
        
        except Exception as e:
            logger.error(f"Error en la función rethink_instruction_list() del planning_expert: {e}")
            raise

    def rethink_subtask_list(self, action_expert_feedback, SOM):
        logger.info("toca repensar la lista de subtareas")
        try:
            prompt = f"""Now that i finished the subtask lets rethink the REST of subtasks. This is what i did: {action_expert_feedback} and take also into account the feedback mentioned in the previous message.
            Taking this into account can you rethink the rest this subtask? Don't edit the task that are already done. Give me this response separating each subtask by a semicolon ';'.
            Remember this is the main task: {self.main_task}"""
            response = self.chat.send_message([prompt, SOM])
            logger.info(f"Esta es la nueva lista de subtareas: {response.text}\n")
            self.subtask_list = self.parse_subtask_response(response.text)

        
        except Exception as e:
            logger.error(f"Error en la función rethink_subtask_list() del planning_expert: {e}")
            raise
    
    def think_instruction_list(self, SOM):
        logger.info("descomponiendo la subtarea en instrucciones")
        try:
            prompt = f"think about the instructions to acomplish this task: {self.subtask_list[self.current_subtask]}"
            response = self.chat.send_message([prompt, SOM])
            logger.info(f"estas son las instrucciones que ha creado {response.text}")
            return response.text
        
        except Exception as e:
            logger.error(f"Error en la función think_instruction_list() del planning_expert: {e}")
            raise
        
    
    def predict(self, action_expert_feedback: str, reflection_expert_feedback: str, SOM):
        
        # mira si la tarea se ha terminado
        finished = self.decide_if_task_finished(reflection_expert_feedback)

        # si no se ha terminado se tiene que repensar las instructions
        if not finished:
            instruction_list = self.rethink_instruction_list(action_expert_feedback, SOM)
            return self.subtask_list[self.current_subtask], instruction_list
        
        if not self.first_iter:
            # si se ha terminado toca repensar el resto de subtareas
            self.rethink_subtask_list(action_expert_feedback, SOM)
            self.current_subtask += 1
            self.first_iter = False

        # en el caso de que haya acabado todas las subtareas no modificará la task list 
        # y por lo tanto current_subtask coincidirá con len(self.subtask_list)
        if len(self.subtask_list) == self.current_subtask:
            return "done", "return 'done'"     
        
        # esto creará la instruction list de la nueva subtarea
        instruction_list = self.think_instruction_list(SOM)
        
        return self.subtask_list[self.current_subtask], instruction_list
        

if __name__ == "__main__":
    pass