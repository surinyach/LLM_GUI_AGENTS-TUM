### Planning Expert

**Iteración 1:** 

- **INPUT: (de OSWorld)** 
    - Tarea general
    - SOM1 (Perception Expert)

Dividir tarea general en subtareas teniendo en cuenta el estado actual proporcionado por el SOM1.

Coger la primera subtarea, y generar pasos para solucionarla, de forma (Sin poner las coordenadas de cada cosa, solo hace el planning):

1. Clica el icono del navegador 
2. Busca en la barra de búsqueda perros
3. Clica una foto de perro
4. ...

Pedir al perception el SOM y los pasos al action expert para que realice la ejecución.

**Iteración N: Error resolviendo la subtarea** 

- **INPUT: (del reflection)** 
    - Plan original de la subtarea.
    - Parte en la que se ha producido el error para que tenga en cuenta lo que ya se ha hecho (si ha descargado ya las imagenes que no lo vuelva a hacer)
    - Error producido para que aprenda de el y no lo vuelva a cometer.  
    - Solución al error pensada por el reflection expert con ayuda del error handling expert.
    - Pedir al memory expert la lista de errores y soluciones pasadas

Replanificar la subtarea para terminarla teniendo en cuenta el contexto en el que actualmente se encuentra.

Corrección de la subtarea N: 

1. Cierra la pestaña del explorador de tareas
2. Clica en el icono del navegador
3. Clica una foto de perro
4. ... 

Pedir el SOM al perception y pasar los nuevos pasos al action expert para que realice la ejecución.

**Iteración N: Subtarea anterior satisfactoriamente resuelta:**

El reflection expert confirma que el action expert ha terminado todos los pasos de la subtarea anterior de forma correcta.

- **INPUT: (del reflection)** 
    - SOM (Perception expert)
    - (El plan anterior ya lo tiene en la memoria)
    - (Tiene los errores cometidos en la creación de las 
    anteriores instrucciones)
    - Verificación por parte del reflection system de que se ha llevado a cabo de forma correcta la anterior subtarea.

Revisa el plan anterior (subtareas creadas) teniendo en cuenta el estado actual de la máquina (mirando el SOM) y la tarea general. Realiza ajustes en las subtareas si considera que el camino no es correcto, o si se puede hacer alguna adaptación para mejorarlo.

Con las nuevas (o las originales si ha decidido no hacer cambios), genera las intrucciones para la subtarea correspondiente (Por ejemplo si se ha completado la subtarea1, entonces se pone a resolver la subtarea2 y así consecutivamente de manera lineal hasta que se completa la tarea general).

Instrucciones de la subtarea N:

1. Abre un documento de texto
2. Clica en el insertor de imagenes dentro del documento
3. Selecciona una imagen de un perro
4. Guarda el documento
5. ...

Enviar nuevamente las instrucciones al action expert para que lleve a cabo las acciones.

### Action Expert

**Iteración N: Inicio de la ejecución**

- **INPUT: (del planning)** 
    - Conjunto de intrucciones para resolver una subtarea generadas por el planning expert.
    - SOM con los contenidos de la pantalla.

Analiza los contenidos del SOM para llevar a cabo la primer instrucción instrucciones, realiza las acciones en la máquina virtual (Con código pyautogui o similar, generando el mismo el código o llamando las herramientas correspondientes).

**Iteración N: Paso N de la ejecución**

- **INPUT:**  
    - Nuevo SOM después de generar el paso N-1 de las instrucciones (Obtenido del Perception Expert). 

Nuevamente analiza los contenidos de SOM más reciente para llevar a cabo la siguiente instrucción, realizando las acciones en la máquina virtual.

**Gestion de errores!!**

Para cada una de las resoluciones de las instrucciones, es posible que el proceso falle por errores en la ejecución de instrucciones anteriores. En ese caso:

- Se pasa el SOM actual, la instrucción donde se ha producido el error y el conjunto global de las intrucciones al reflection expert para que encuentre una solución.

### Reflection Expert

**Iteración N: Final de tarea satisfactiorio**

- **INPUT:** 
    - SOM
    - Indicador de que se ha completado la tarea correctamente (Una frase, un booleano, cualquier indicador)
    - Lista de instrucciones que se han presuntamente llevado a cabo de manera satisfactoria.

Verificar que las instrucciones de la subtarea se han llevado a cabo de manera satisfactoria.

Informar al planning system del resultado. Si no es correcta la ejecución para que vuelva a planificar la subtarea y si es correcta la ejecución para que pase a la siguiente subtarea.

**Iteración N: Error en alguna instrucción de la subtarea**

- **INPUT:** 
    - SOM
    - Instruccion que ha fallado
    - Todas las instrucciones de la subtarea
    - Pedir al memory expert la lista de errores y soluciones pasadas

Descubrir porque ha fallado la instrucción. Diagnostico de posibles causas del fallo y potenciales soluciones. Puede hacer uso del error handling expert en caso de ser un error complejo.

### Perception Expert

**Iteración N: Obtener el SOM de una screenshot del benchmark**

Obtiene la Screenshot del estado actual de la máquina del benchmark y genera el SOM correspondiente a la imagen. Para hacerlo se utiliza el modelo OmniparserV2.

Proporciona la Screenshot con el SOM y el contenido de la misma en formato JSON a los demás expertos.

### Error Handling Expert

**Iteracion N: Error complejo que el reflection es incapaz de resolver por si mismo**

- INPUT: Error a solucionar enviado por el reflection expert.

Proporciona una solucion al error recibido como input, después le transmite el conocimiento al reflection system, siendo capaz este último de finalizar sus tareas.

### Memory Expert

**Iteración N:** 

- **INPUT: (del reflection)** 
    - errores y soluciones proporcionado por el reflection

Guardar en una estructura estos errores y soluciones para que el planning y el reflection puedan acceder a ellos.

