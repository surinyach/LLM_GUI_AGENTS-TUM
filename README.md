# 🧠 Barry Agent: LLM-Driven GUI Agent for OSWorld

**Barry Agent** is an agent powered by Large Language Models (LLMs), designed to automate graphical (GUI) tasks in real environments. Our architecture is adapted to the **[OSWorld](https://github.com/xlang-ai/OSWorld)** benchmark, where we evaluate the agent's performance on a representative set of tasks from different categories.

This project was developed as part of the practical course _**Development of LLM-driven GUI Agents**_, belonging to the academic exchange program of the **Computer Engineering degree at FIB (UPC)**, during the semester at **Technische Universität München (TUM)**, in Munich, Germany.

---

## 🚀 Table of Contents

- [🔧 System Requirements](#-system-requirements)
- [📦 Installation](#-installation)
  - [1. OSWorld Installation](#1-osworld-installation)
  - [2. Barry Agent Installation](#2-barry-agent-installation)
  - [3. Perception Expert Configuration](#3-perception-expert-configuration)
- [⚙️ Execution](#️-execution)
- [🔄 Reproducing Evaluated Tasks](#-reproducing-evaluated-tasks)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [👥 Authors](#-authors)

---

## 🔧 System Requirements

- Python 3.10.18 (we recommend using [Anaconda](https://www.anaconda.com/))
- Docker
- Access to the [Gemini API](https://ai.google.dev/)
- ✅ **Recommended:** machine with GPU for efficient OmniParser execution (although it's possible to run it with CPU, with significantly lower performance)

---

## 📦 Installation

### 1. OSWorld Installation

1. Go to the official [OSWorld](https://github.com/xlang-ai/OSWorld) repository.
2. Carefully follow their [installation instructions](https://github.com/xlang-ai/OSWorld#-installation).
3. Once the installation is complete, the environment will be ready to integrate with Barry Agent.

---

### 2. Barry Agent Installation

1. Copy the contents of this repository to the OSWorld root directory.
2. Make sure you have the necessary dependencies installed:

```bash
pip install -r requirements.txt  # Run in the Barry Agent directory
pip install -r requirements.txt  # Also in OSWorld
```

---

### 3. Perception Expert Configuration

Barry Agent uses a Perception Expert based on the Set-Of-Mark model, employing OmniParser to extract visual elements, coordinates, and descriptions from the GUI.

#### 🐳 Docker Configuration

We have prepared a Dockerfile with the necessary image. You can run it locally with:

```bash
docker build -t omniparser-server .
docker run -p 8000:8000 omniparser-server
```

#### ☁️ Cloud Usage (DockerHub)

You can also directly use our image hosted on DockerHub:

- 📦 `surinyach/omniparser-server`

Commands to upload your own image:

```bash
docker build -t omniparser-server .
docker tag omniparser-server DOCKERHUB_USERNAME/omniparser-server:latest
docker login
docker push DOCKERHUB_USERNAME/omniparser-server:latest
```

#### 💻 Environment Recommendation

Although OmniParser can run on CPU, we strongly recommend using a machine with GPU for optimal performance. In our case, we used RunPod with the following configuration:

- **30 GB Disk**
- **20 GB Pod Volume**
- **GPU:** Nvidia RTX A4000
- **35 GB RAM**

**Recommended template:** `docker.io/surinyach/omniparser-server:latest`

> ⚠️ **Note:** By default, the server runs on port 8000. You can change it by editing the last line of the Dockerfile adding: `--port [PORT]`.

---

## ⚙️ Execution

1. Create a `.env` file inside the `barry_agent` directory with the following content:

```env
GEMINI_API_KEY=your_gemini_api_key
OMNIPARSER_SERVER_URL=https://YOURDOMAIN:8000/
```

2. Run the agent with the following basic command:

```bash
python run_barry.py --model gemini-2.5-flash --observation_type screenshot --action_space pyautogui
```

3. To launch specific tasks, for example to maximize the volume:

```bash
python run_barry.py \
  --model gemini-2.5-flash \
  --observation_type screenshot \
  --action_space pyautogui \
  --domain os \
  --task 28cc3b7e-b194-4bc9-8353-d04c0f4d56d2
```

---

## 🔄 Reproducing Evaluated Tasks

To reproduce the exact tasks that were evaluated during the development and testing of Barry Agent, you can find the complete list of task identifiers and their corresponding domains in the **Final Report** located in the `Resources/` directory of this repository.

The appendix of the Final Report contains detailed information about all evaluated tasks, including their identifiers and domain categories. To run any of these specific tasks, use the command structure shown in the execution section above, replacing the `--domain` and `--task` parameters with the desired values from the report.

**Example command structure:**

```bash
python run_barry.py \
  --model gemini-2.5-flash \
  --observation_type screenshot \
  --action_space pyautogui \
  --domain [DOMAIN_FROM_REPORT] \
  --task [TASK_ID_FROM_REPORT]
```

---

## 🛠️ Troubleshooting

### Missing Logs Directory

**Issue:** Execution fails with directory-related errors.

**Solution:** Make sure to create a `logs` directory within the OSWorld root directory before running Barry Agent. The agent requires this directory to store execution logs and will fail if it doesn't exist.

```bash
mkdir logs  # Run this command in the OSWorld root directory
```

### OmniParser Server URL Configuration

**Issue:** Execution fails with URL-related errors when connecting to the OmniParser server.

**Solution:** Ensure that the `OMNIPARSER_SERVER_URL` in your `.env` file does **NOT** end with a trailing slash (`/`). This will cause connection errors.

**❌ Incorrect:**
```env
OMNIPARSER_SERVER_URL=https://YOURDOMAIN:8000/
```

**✅ Correct:**
```env
OMNIPARSER_SERVER_URL=https://YOURDOMAIN:8000
```

### OmniParser Server Connectivity

**Issue:** Execution fails during the perception expert stage with connection timeouts or unreachable server errors.

**Solution:** Verify that your OmniParser server is running and accessible before executing Barry Agent. You can test connectivity using:

```bash
curl -I https://YOURDOMAIN:8000/probe
```

Make sure your server is properly deployed and the specified port is open and accessible from your execution environment.

---

## 👥 Authors

- 🧑‍💻 [Santi Oliver](https://github.com/surinyach)
- 🧑‍💻 [Víctor De Lamo](https://github.com/VictorDeLamo)