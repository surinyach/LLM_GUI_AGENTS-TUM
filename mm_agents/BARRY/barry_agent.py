import os
import logging
from typing import Dict, List, Tuple, NamedTuple
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from io import BytesIO
from .action_expert import ActionExpert
from .planning_expert import PlanningExpert
from .reflection_expert import ReflectionExpert
from .error_expert import ErrorExpert

from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

logger = logging.getLogger("desktopenv.agent")

class BarryAgent:
    def __init__(self, model: str = "gemini-2.0-flash", observation_type: str = "screenshot", action_space: str = "pyautogui"):
        """
        Inicializa el agente con la configuración para interactuar con OSWorld y la API de Gemini.
        """
        # Cargar variables de entorno desde .env
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            logger.error("GEMINI_API_KEY no encontrada en el archivo .env")
            raise ValueError("GEMINI_API_KEY no encontrada en el archivo .env")

        # Configurar la API de Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model)

        # Configurar parámetros del entorno
        self.observation_type = observation_type
        self.action_space = action_space
        
        # Historial para el agente
        self.trajectory_length = 0
        self.max_trajectory_length = 50
        self.call_user_count = 0
        self.call_user_tolerance = 3
        

        # PERCEPTION SYSTEM
        self.SOM = ""
        

        self.main_task = ""
        self.first_iteration = True

        self.action_expert = ActionExpert() 
        self.planning_expert = PlanningExpert()
        self.reflection_expert = ReflectionExpert()
        self.error_expert = ErrorExpert()
        
        self.graph = None
        self.graph_state = {} 

        

        # LANG GRAPH

        # STATE --------------------------------------------------------------------

        class State(TypedDict):
            #intruction_list: str    # no hace falta guardarla porque se la pasa directamente cuando las crea
            action_expert_feedback: str
            reflection_expert_feedback: str
            info_for_error_expert: str
            error_expert_feedback: str
            new_instruction_list: bool

            done: bool
            osworld_action: str
        

        graph_builder = StateGraph(State)

        # NODES -----------------------------------------------

        def start_router(state: State):
            logger.info("estoy en el start_router")

            if self.first_iteration:
                return {"next": "planning_expert"}
            return {"next": "action_expert"}
        
        def planning_expert(state: State):
            if self.first_iteration:
                self.planning_expert.save_main_task(self.main_task)
                self.first_iteration = False
            
            instruction_list, subtask = self.planning_expert.predict(
                state.get("action_expert_feedback",""),
                state.get("reflection_expert_feedback",""), 
                self.SOM)

            if instruction_list == "done": # significa que ya no hay nada más que hacer
                return {
                "done": True,
                }
            else: # pongo else para mejorar la legibilidad
                self.action_expert.add_new_instructions(subtask, instruction_list)
                self.reflection_expert.save_instruction_list(subtask, instruction_list)
                return {
                "done": False,
                }

        



        def action_expert(state: State):
            if state.get("done", False):
                return {"osworld_action": "done"}

            osworld_action = self.action_expert.predict(self.SOM)

            if osworld_action.startswith("error:"):
                return {"execution_error": osworld_action,
                        "osworld_action": None, 
                        "new_instruction_list": False}
            
            return {"osworld_action": osworld_action, 
                    "new_instruction_list": False}
        
        def action_router(state: State):
            condition = not state.get("done", False) and state.get("execution_error", "")
            if condition: # "" en python es falsy y cualquier otro string será true
                return {"next": "reflection_expert"}
            
            return {"next": "end"} # para que el predict devuelva la acción y se pueda volver a llamar al predict después.
        


        def reflection_expert(state: State):
            
            if state.get("error_expert_feedback", None) is None:
                info_for_error_expert, reflection_expert_feedback = self.reflection_expert.predict(
                    state.get("execution_error", ""), # si no hay execution error es que simplemente ha acabado su lista
                    state.get("action_expert_feedback", ""), # las instruction list ya las ha guardado el planning cuando crea nuevas
                    self.SOM) 
                

                if info_for_error_expert is not None:
                    return {"info_for_error_expert": info_for_error_expert}
            else:
                # no le pasamos nada más porque ya lo tiene guardado en el historial del chat
                reflection_expert_feedback = self.reflection_expert.predict_with_error_expert_feedback(state["error_expert_feedback"], self.SOM)
            
            return {"reflection_expert_feedback": reflection_expert_feedback} # si ya ha acabado correctamente este feedback dirá que ha acabado correctamente


        def reflection_router(state: State):
            if state.get("info_for_error_expert", None) is not None:
                return {"next": "error_expert"}
            return {"next": "planning_expert"}




        def error_expert(state: State):
            error_expert_feedback = self.error_expert.predict(state["info_for_error_expert"], self.SOM)
                
            return {"error_expert_feedback": error_expert_feedback,
                    "info_for_error_expert": None} # lo dejamos en blanco para la siguiente vez
        

        # EDGES ----------------------------------------------------------

        graph_builder.add_node("start_router", start_router)
        graph_builder.add_node("planning_expert", planning_expert)
        graph_builder.add_node("action_expert", action_expert)
        graph_builder.add_node("action_router", action_router)
        graph_builder.add_node("reflection_expert", reflection_expert)
        graph_builder.add_node("reflection_router", reflection_router)
        graph_builder.add_node("error_expert", error_expert)

        graph_builder.add_edge(START, "start_router")
        graph_builder.add_conditional_edges(
            "start_router",
            lambda state: state.get("next"),
            {"planning_expert": "planning_expert", "action_expert": "action_expert"}
        )
        graph_builder.add_edge("planning_expert", "action_expert")
        graph_builder.add_edge("action_expert", "action_router")
        graph_builder.add_conditional_edges(
            "action_router",
            lambda state: state.get("next"),
            {"reflection_expert": "reflection_expert", "end": END}
        )
        graph_builder.add_edge("reflection_expert", "reflection_router")
        graph_builder.add_conditional_edges(
            "reflection_router",
            lambda state: state.get("next"),
            {"planning_expert": "planning_expert", "error_expert": "error_expert"}
        )
        graph_builder.add_edge("error_expert", "reflection_expert")

        # COMPILE ---------------------------------------------------

        self.graph = graph_builder.compile()





    def process_observation(self, obs: Dict) -> Image.Image:
        """
        Procesa la captura de pantalla del entorno para convertirla a una imagen PIL compatible con Gemini.
        """
        if self.observation_type not in ["screenshot", "screenshot_a11y_tree"]:
            logger.error(f"observation_type debe ser 'screenshot' o 'screenshot_a11y_tree', recibido: {self.observation_type}")
            raise ValueError(f"observation_type no soportado: {self.observation_type}")

        if "screenshot" not in obs:
            logger.error("No se encontró 'screenshot' en la observación")
            raise ValueError("No se encontró 'screenshot' en la observación")

        try:
            screenshot = obs["screenshot"]
            # Manejar si screenshot es str
            if isinstance(screenshot, str):
                screenshot = screenshot.encode('utf-8')
            # Convertir a imagen PIL
            image = Image.open(BytesIO(screenshot))
            return image
        except Exception as e:
            logger.error(f"Error al procesar el screenshot: {e}")
            raise


    def predict(self, instruction: str, obs: Dict) -> Tuple[str, List[str]]:
        """
        Envía la captura de pantalla y la instrucción a Gemini para generar acciones pyautogui.
        Retorna una tupla con la respuesta de Gemini y la lista de acciones con estados apropiados.
        """
        logger.info("Predict de barry agent")
        self.trajectory_length += 1
        
        if self.trajectory_length > self.max_trajectory_length:
            logger.warning(f"Trayectoria excede el límite máximo de {self.max_trajectory_length} pasos")
            return "Maximum trajectory length exceeded", ["FAIL"]
        
        try:
            self.SOM = self.process_observation(obs)
            
            # Si es la primera iteración copiamos la tarea y la añadimos al historial
            if self.first_iteration:
                self.main_task = instruction
                # Inicializa el estado del grafo con la tarea principal si es la primera vez
                # LangGraph se encargará de inicializar instruction_list en planning_expert
                self.graph_state = {
                    "action_expert_feedback": "",
                    "reflection_expert_feedback": "",
                    "info_for_error_expert": "",
                    "error_expert_feedback": "",
                    "new_instruction_list": True,

                    "done": False,
                    "osworld_action": "",
                }
            
            # si no es la primera iteración ya tienes el grafo guardado de antes

            final_state = self.graph.invoke(self.graph_state)
            self.graph_state = final_state
            osworld_action_to_return = final_state.get("osworld_action")
            is_done = final_state.get("done", False)

            if is_done and osworld_action_to_return == "done": # doble comprobación aun que con una ya sería suficiente
                logger.info("BarryAgent: Tarea finalizada con éxito.")
                return "se acabo lo que se daba", ["DONE"]

            if osworld_action_to_return:
                logger.info(f"BarryAgent: Acción decidida por el agente: '{osworld_action_to_return}'")
                return "esta es la siguiente acción", [osworld_action_to_return]
            else:
                logger.warning("BarryAgent: El grafo no produjo una acción de OSWorld válida en esta iteración.")
                return "FAIL: No OSWorld action generated in this cycle.", ["FAIL"]

            
        except Exception as e:
            logger.error(f"Error al procesar la solicitud a Gemini: {e}")
            return f"Error: {e}", ["FAIL"]


    def reset(self, runtime_logger):
        """
        Reinicia el estado del agente para una nueva tarea.
        """
        self.trajectory_length = 0
        self.call_user_count = 0
        self.first_iteration = True
        logger.info("Agente reiniciado")



if __name__ == "__main__":
    """
    Bloque de prueba local para el agente.
    """
    try:
        # Crear una instancia del agente
        agent = BarryAgent()
        
        # Reiniciar el agente
        agent.reset(None)
        
        # Instrucción de prueba
        test_instruction = "Listar los archivos en el directorio actual"
        
        # Simular observación de prueba (necesitarías un screenshot real)
        test_obs = {
            "screenshot": b"",  # Aquí iría los bytes del screenshot real
        }
        
        print("Agente inicializado correctamente")
        
        
        # Para probar realmente, necesitarías cargar un screenshot real
        # response, actions = agent.predict(test_instruction, test_obs)
        # print(f"Respuesta: {response}")
        # print(f"Acciones: {actions}")
        
    except Exception as e:
        print(f"Error en la prueba: {e}")
        print("Asegúrate de tener configurado GEMINI_API_KEY en tu archivo .env")