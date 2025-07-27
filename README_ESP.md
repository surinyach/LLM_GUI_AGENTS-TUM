# 🧠 Barry Agent: LLM-Driven GUI Agent for OSWorld

**Barry Agent** es un agente impulsado por modelos de lenguaje de gran tamaño (LLMs), diseñado para automatizar tareas gráficas (GUI) en entornos reales. Nuestra arquitectura está adaptada al benchmark **[OSWorld](https://github.com/xlang-ai/OSWorld)**, donde evaluamos el desempeño del agente en un conjunto representativo de tareas de distintas categorías.

Este proyecto fue desarrollado como parte del curso práctico _**Development of LLM-driven GUI Agents**_, perteneciente al programa de intercambio académico del **grado en Ingeniería Informática de la FIB (UPC)**, durante el semestre en la **Technische Universität München (TUM)**, en Múnich, Alemania.

---

## 🚀 Tabla de contenidos

- [🔧 Requisitos del sistema](#-requisitos-del-sistema)
- [📦 Instalación](#-instalación)
  - [1. Instalación de OSWorld](#1-instalación-de-osworld)
  - [2. Instalación de Barry Agent](#2-instalación-de-barry-agent)
  - [3. Configuración del Perception Expert](#3-configuración-del-perception-expert)
- [⚙️ Ejecución](#️-ejecución)
- [📌 Notas adicionales](#-notas-adicionales)
- [📫 Contacto](#-contacto)

---

## 🔧 Requisitos del sistema

- Python 3.10.18 (recomendamos usar [Anaconda](https://www.anaconda.com/))
- Docker
- Acceso a la [Gemini API](https://ai.google.dev/)
- ✅ **Recomendado:** máquina con GPU para ejecución eficiente de OmniParser (aunque también es posible correrlo con CPU, con rendimiento significativamente inferior)

---

## 📦 Instalación

### 1. Instalación de OSWorld

1. Dirígete al repositorio oficial de [OSWorld](https://github.com/xlang-ai/OSWorld).
2. Sigue cuidadosamente sus [instrucciones de instalación](https://github.com/xlang-ai/OSWorld#-installation).
3. Una vez completada la instalación, el entorno estará listo para integrarse con Barry Agent.

---

### 2. Instalación de Barry Agent

1. Copia el contenido de este repositorio en el directorio raíz de OSWorld.
2. Asegúrate de tener instaladas las dependencias necesarias:

```bash
pip install -r requirements.txt  # Ejecutar en el directorio de Barry Agent
pip install -r requirements.txt  # También en el de OSWorld
```

---

### 3. Configuración del Perception Expert

Barry Agent utiliza un Perception Expert basado en el modelo Set-Of-Mark, empleando OmniParser para extraer elementos visuales, coordenadas y descripciones desde la GUI.

#### 🐳 Configuración con Docker

Hemos preparado un Dockerfile con la imagen necesaria. Puedes ejecutarlo localmente con:

```bash
docker build -t omniparser-server .
docker run -p 8000:8000 omniparser-server
```

#### ☁️ Uso en la nube (DockerHub)

También puedes usar directamente nuestra imagen alojada en DockerHub:

- 📦 `surinyach/omniparser-server`

Comandos para subir tu propia imagen:

```bash
docker build -t omniparser-server .
docker tag omniparser-server DOCKERHUB_USERNAME/omniparser-server:latest
docker login
docker push DOCKERHUB_USERNAME/omniparser-server:latest
```

#### 💻 Recomendación de entorno

Aunque OmniParser puede ejecutarse en CPU, recomendamos fuertemente usar una máquina con GPU para un desempeño óptimo. En nuestro caso utilizamos RunPod con la siguiente configuración:

- **30 GB Disk**
- **20 GB Pod Volume**
- **GPU:** Nvidia RTX A4000
- **35 GB RAM**

**Plantilla recomendada:** `docker.io/surinyach/omniparser-server:latest`

> ⚠️ **Nota:** Por defecto, el servidor se ejecuta en el puerto 8000. Puedes cambiarlo editando la última línea del Dockerfile agregando: `--port [PUERTO]`.

---

## ⚙️ Ejecución

1. Crea un archivo `.env` dentro del directorio `barry_agent` con el siguiente contenido:

```env
GEMINI_API_KEY=tu_clave_gemini_api
OMNIPARSER_SERVER_URL=https://TUDOMINIO:8000/
```

2. Ejecuta el agente con el siguiente comando básico:

```bash
python run_barry.py --model gemini-2.5-flash --observation_type screenshot --action_space pyautogui
```

3. Para lanzar tareas específicas, por ejemplo subir el volumen al máximo:

```bash
python run_barry.py \
  --model gemini-2.5-flash \
  --observation_type screenshot \
  --action_space pyautogui \
  --domain os \
  --task 28cc3b7e-b194-4bc9-8353-d04c0f4d56d2
```
---

## 👥 Autores

- 🧑‍💻 [Santi Oliver](https://github.com/surinyach)
- 🧑‍💻 [Víctor De Lamo](https://github.com/VictorDeLamo)