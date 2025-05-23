# Arrythmia-Classifier
Heart arrhythmias have been one of the leading causes of deaths globally. The electrocardiogram 
(ECG) is a standard non-invasive test used to monitor the heart’s activity and diagnose any potential 
heart diseases. The analysis of signals outputted from the ECG are visually analysed by clinicians. 
However, depending on the skill of the clinician or other external factors, there are some challenges 
posed when accurately diagnosing for heart arrhythmias. This project aims to build a deep learning 
classifier capable of identifying and classifying arrhythmias from ECG signals, as well as evaluating 
the effectiveness of three deep learning models, CNNs, LSTMs and a  hybrid CNN-LSTM model 
under different data structures. 

## Project Flowchart
![image](https://github.com/user-attachments/assets/e585859f-906d-4c69-b63f-428ce3e05cf1)

## Starting the Project

### 1. Downloading the Dataset

Link to download dataset (MIT-BIH Arrhythmia Database): https://www.physionet.org/content/mitdb/1.0.0/
Press the 'Download the ZIP file' to download all the records. Extract the entire zip folder and place them into a designated folder on your device.
Please ensure that you have one folder created for this entire project, as it will involve multiple CSV files created.
### 2. Setting up a GPU in your device

Follow the steps below if your device's graphic processing unit (GPU) has not been configured yet, or skip it if your device doesn't have a GPU.
**A. Install Nvidia GPU Driver**

Link to download Nvidia Driver: https://www.nvidia.com/en-us/drivers/
Select your GPU model and operating system.
Download the appropriate driver and follow the installation instructions.
Restart your device after the driver is installed.
To validate if your dirver has been successfully installed, open Command Prompt and run: nvidia-smi
This will show information about your GPU, memory usage, and other details. The image below shows an example on my device (my devices GPU is GTX1650).
![image](https://github.com/user-attachments/assets/633c4e9d-fa76-4200-920d-2cfad90e9c04)

**B. Install CUDA Toolkit**

CUDA Installation: https://developer.nvidia.com/cuda-downloads
Select your operating system and the appropriate version of CUDA (I used CUDA 11.8).
Download the installer and run it
Follow the installation instructions, and ensure you choose the default installation (includes the necessary libraries and compiler)

**C. Install Conda and Create a Virtual Environment**

Link to download Anaconda: https://www.anaconda.com/download (skip this step if you already have Anaconda installed)
Install Anaconda by following the instructions on the website
Once it is downloaded or you already ahve it, proceed to open Anaconda Prompt and create a new environment for PyTorch: conda create --name test(test is the name of the virtual envrironment) conda activate test
This creates a virtual environment called 'test' and activates it

**D. Install PyTorch with CUDA Support**

After activating the virtual environment, install PyTorch with CUDA support
Run the following prompt in the Anaconda Prompt (while the virtual environment is activated): conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
This will install PyTorch, TorchVision and CUDA 11.8 support
You can verify its installation to check if CUDA is enabled in PyTorch, its output should be like the image below
![image](https://github.com/user-attachments/assets/a7d16fa5-cac6-49ed-8bc1-0fc7003f3950)

**E. Install Other required libraries within the virtual environment**

To install other libraries, you can install them using 'pip' within the virtual environment pip install pandas numpy matplotlib scikit-learn seaborn

**F. Set up Jupyter Notebook with the Correct Kernel**

This project will be run on Jupyter Notebook excpet for the web application, ensure that this is installed
Install ipykernel in the virtual environment: pip install ipykernel
Add the virtual environment as a Jupyter kernel: python -m ipykernel install --user --name=test --display-name "Experiment"
After installing the kernel, you can start Jupyter Notebook by running: jupyter notebook (Or alternatively, opening the jupyter notebook application itself)

### 3. Code Base

There are a total of 4 Jupyter Notebook and 1 Python file:
'Dataset and Preprocessing.ipynb': Dataset preprocessing, exploratory data analysis of the dataset and dataset creation (Note: this can be run under your default environment, it doesn't necessarily have to be run under the new virtual environemnt setup for the GPU)
'Experiments - Dataset A.ipynb': Experiments run on Dataset A (Individual Beats)
'Experiments - Dataset B.ipynb': Experiments run on Dataset B (5-beat Segments)
'Experiments - Dataset C.ipynb': Experiments run on Dataset C (5-second Segments)
'web_app.py': Web application for model deployment.
NOTE: Please ensure that you run these files in order (start from number 1 then to 5 ascendingly)

### 4. Web Application

There is a web application included in this project developed using Steamlit.
To set up Streamlit, run the following command on your Command Prompt: pip install streamlit
Once installed, open the web_app.py file and navigate to the directory containing this file on Visual Studio Code.
Open a terminal and enter this prompt: streamlit run web_app.py. This will open the application on your browser.
Before running the web application, please ensure that you have run all the cells in 'Experiment 1 - Dataset A.ipynb'. There is a cell which saves the model into 'hybrid_model.pth', which is the model used in the application.
On the website there will be a prompt to upload a file, note that the files accepted for the model to work are limited to CSV fie types only.
After running the 'Dataset and Preprocessing.ipynb' file, there will be CSV files of the ECG records. Upload these files onto the application. (note that only one file can be uploaded at a time)

