from mm_agents.BARRY.perception import PerceptionExpert
import base64

# Transform the .png image into bytes format
with open("mm_agents/BARRY/screenshot.png", "rb") as f:
    image_bytes = f.read()

perception = PerceptionExpert()

# Store the screenshot inside the PerceptionExpert class in base64 format
perception.store_screenshot(image_bytes)

# Generate the SOM of the stored screenshot
perception.process_screenshot()

# Parse the resulting screenshot
result = perception.get_som_screenshot()
with open("annotated.png", "wb") as f:
    f.write(base64.b64decode(result))

# Print the recieved description
print("Parsed elements:")
result = perception.get_som_description()
for el in result:
    print(el)