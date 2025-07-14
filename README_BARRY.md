pip install -r mm_agents\BARRY\requirements.txt 

para ejecutar una tarea usa este comando:

python run_barry.py --model gemini-2.0-flash --observation_type screenshot --action_space pyautogui --domain os --task 28cc3b7e-b194-4bc9-8353-d04c0f4d56d2

puedes elegir la tarea en la carpeta evaluation_examples/examples

--domain <aquí pones la sub carpeta de la cual hayas elegido la tarea>
--task <aquí pones el id de la tarea>