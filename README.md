# 🌤️ Weather Forecasting with TensorFlow & SLURM

A simple machine learning project for predicting weather conditions using historical data. The model is trained using Python and TensorFlow, and jobs are scheduled with **SLURM**, a popular HPC workload manager.

---

## 📊 Dataset

- **File:** `weather.csv`  
- **Content:** Includes temperature, humidity, wind speed, and other meteorological features used for training the model.

---

## 🚀 Quick Start

### 🔧 Prerequisites

- Python 3.9+
- TensorFlow
- SLURM workload manager
- Optional: Virtual environment (`venv`)

### 🛠️ Installation

```bash
git clone https://github.com/your-username/Demo-ML-TF-Slurm.git
cd Demo-ML-TF-Slurm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  # You can create this based on your project
🧠 Training the Model
To train the model, use SLURM to submit the job script:

bash
Copy
Edit
sbatch train_job.slurm
📄 train_job.slurm
bash
Copy
Edit
#!/bin/bash
#SBATCH --job-name=weather-train
#SBATCH --output=train_output.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=512M
#SBATCH --time=00:05:00
#SBATCH --partition=debug

module load python/3.9
cd /home/<your-username>/Demo-ML-TF-Slurm
python3 model.py
📌 Be sure to update the working directory path in the SLURM script.

🔍 Check Job Status
bash
Copy
Edit
squeue -u $USER
📁 Output
Once completed, check train_output.log:

bash
Copy
Edit
cat train_output.log
🔮 Making Predictions
After training completes, use:

bash
Copy
Edit
python3 predict.py
This script:

Loads the trained model

Accepts new input data

Prints predictions

📌 Notes
Ensure the SLURM node is not in a DRAIN state before job submission.

Memory and CPU values in the SLURM script should align with your cluster configuration.

