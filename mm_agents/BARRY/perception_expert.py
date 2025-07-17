import base64
import requests
import os
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
        self.omniparser_server = os.getenv("OMNIPARSER_SERVER_IP")

        # Local variables
        self.screenshot = ""
        self.som_screenshot = ""
        self.som_description = ""

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
            self.som_screenshot = result["som_image_base64"]
            self.som_description = result["parsed_content_list"]

        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Error connecting to Omniparser server: {e}")
        
    def get_som_screenshot(self):
        """
        Provides the SOM screenshot to the caller.
        In order to get the desired screenshot first is necessary to run the store_screenshot
        and process_screenshot functions.

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


