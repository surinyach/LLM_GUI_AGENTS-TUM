from mm_agents.BARRY.perception_expert import PerceptionExpert
from PIL.Image import Image as PIL_Image
import base64

# Transform the .png image into bytes format
with open("mm_agents/BARRY/screenshot.png", "rb") as f:
    image_bytes = f.read()

perception = PerceptionExpert()

# Store the screenshot inside the PerceptionExpert class in base64 format
perception.store_screenshot(image_bytes)

# Generate the SOM of the stored screenshot
perception.process_screenshot()

result_image = perception.get_som_screenshot()
if isinstance(result_image, PIL_Image):
    # Save the PIL Image directly to a PNG file
    result_image.save("annotated.png", "PNG")
    print("Annotated image saved as annotated.png")
else:
    print("Error: The returned object is not a PIL Image.")

# Print the recieved description
print("Parsed elements:")
result = perception.get_som_description()
print(result)