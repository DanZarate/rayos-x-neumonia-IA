from roboflow import Roboflow

# Initialize Roboflow and download the Chest X-Ray dataset
rf = Roboflow(api_key="APMubwY7a5CpLo2HJ8Yo")
project = rf.workspace("mohamed-traore-2ekkp").project("chest-x-rays-qjmia")

# Download version 4 in folder format (versions 1-3 had API issues)
version = project.version(4)
dataset = version.download("folder")

                