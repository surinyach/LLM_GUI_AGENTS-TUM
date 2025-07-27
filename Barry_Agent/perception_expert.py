import base64
import requests
import os
from PIL import Image
from io import BytesIO
import io
from dotenv import load_dotenv

class PerceptionExpert:
    def __init__(self):
        """
        Initializes the PerceptionExpert with necessary environment variables (Omniparser server URL).

        The Perception Expert is in charge of processing the screenshots given by the OSWorld benchmark,
        generating a SetOfMark(SOM) and a description of each marked element in JSON format.

        To accomplish that it communicates with an API server which holds an Omniparser (https://github.com/microsoft/OmniParser/)
        model, which carries the task and returns the highlighted screenshot and its description.
        """

        load_dotenv()
        self.omniparser_server = os.getenv("OMNIPARSER_SERVER_URL")

        # Local variables
        self.screenshot = ""
        self.som_screenshot = ""
        self.som_description = ""

    # LOCAL FUNCTIONS
    def _format_som_description(self, elements):
        """
        Formats the description of the SOM image to be human-readable instead of having a JSON format.
        
        Args:
            elements(str): Description in JSON format
        
        Returns:
            som_description(str): Description in human-readable content
        """
        formatted_list = []
        for idx, element in enumerate(elements, start=0):
            element_type = element.get('type', 'unknown')
            content = element.get('content', 'no content')
            is_interactive = "Interactive" if element.get('interactivity', False) else "non-interactive"
            bbox = [round(b, 3) for b in element.get('bbox', [])]

            formatted_list.append(
                f"{idx}. {element_type.capitalize()}: '{content}' (Bounding Box: {bbox}, {is_interactive})"
            )
    
        return "\n".join(formatted_list)

    def store_screenshot(self, screenshot):
        """
        Recieves a new screenshot from the OSWorld environment and stores it in the self.screenshot local variable.

        Params:
            screenshot: The screenshot representing the up to date state of the OSWorld machine
        """
        # If the screenshot is a byte string (raw screenshot data), process it into base64
        if isinstance(screenshot, bytes):
            screenshot = base64.b64encode(screenshot).decode('utf-8')
        
        self.screenshot = screenshot

    def process_screenshot(self):
        """
        Processes the screenshot to produce the SOM and the description of its elements.

        To do it, it sends the screenshot to the omniparser-server which is the one who
        makes the computation and returns the results.

        Stores the marked screenshot and the description in the self.som_screenshot and 
        self.som_description respectively.
        """
        url = f"{self.omniparser_server}/parse/"
        payload = {"base64_image": self.screenshot} 
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() # Raise an exception for HTTP Errors
            result = response.json()
            
            # Convert the base64 string to a Pillow Image
            som_image_64 = result["som_image_base64"]
            image_bytes = base64.b64decode(som_image_64)
            self.som_screenshot = Image.open(BytesIO(image_bytes))
            
            self.som_description = self._format_som_description(result["parsed_content_list"])

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error connecting to Omniparser server: {e}")
        
    def get_som_screenshot(self):
        """
        Provides the SOM screenshot to the caller.
        In order to get the desired screenshot first is necessary to run the 'store_screenshot'
        and 'process_screenshot' functions.

        Returns:
            SOM screenshot in base64 format.
        """
        return self.som_screenshot

    def get_som_description(self):
        """
        Provides the SOM description to the caller.
        In order to get the desired description first is compulsory to run the store_screenshot
        and process_screenshot functions.

        Returns:
            SOM description with a JSON format
        """
        return self.som_description
    
    def get_screenshot(self):
        """
        Provides the vanilla screenshot without the SOM to the caller.
        To do so, transforms self.screenshot into a PILLOW image.

        Returns:
            screenshot(PIL Image): The vanilla screenshot in PILLOW format.
        """
        # Decode the base64 string to bytes
        image_bytes = base64.b64decode(self.screenshot)
        # Use io.BytesIO to read 
        image_stream = BytesIO(image_bytes)

        # Open the image using PIL
        pil_image = Image.open(image_stream)
        return pil_image

