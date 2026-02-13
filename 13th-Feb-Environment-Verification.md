
# Data Science Environment Verification

## Milestone Objective

This milestone verifies that the local machine is properly configured for the Data Science sprint.  

The goal is to confirm that:

- Python is installed and working
- Conda environments are functional
- Jupyter Notebook/Lab launches and executes Python correctly
- All tools work together reliably

This verification ensures a stable and reproducible environment before beginning any data science development.

---

## 1. Operating System

```

Windows 11

````


---

## 2. Python Verification

### Python Version

Command used:

```bash
python --version
````

Output:

```
Python 3.11.6
```

### Python REPL Test

Command used:

```bash
python
```

Tested commands inside REPL:

```python
print("Environment verification successful")
2 + 2
```

Output:

```
Environment verification successful
4
```

Conclusion:

* Python launches successfully
* Python executes code correctly
* Installation is stable

---

## 3. Conda Environment Verification

### Conda Version

Command used:

```bash
conda --version
```

Output:

```
conda 23.7.4
```

### List Available Environments

Command used:

```bash
conda env list
```

Sample Output:

```
# conda environments:
#
base                  *  C:\Users\YourName\anaconda3
```

### Environment Activation

Command used:

```bash
conda activate base
```

Verification:

* Terminal prompt reflects active environment `(base)`
* Python runs correctly inside the environment

Conclusion:

* Conda is installed
* Environment activation works correctly
* Python is callable within the Conda environment

---

## 4. Jupyter Verification

### Launch Command

```bash
jupyter notebook
```

OR

```bash
jupyter lab
```

Verification Steps:

* Jupyter opened successfully in browser
* Created a new Python notebook
* Executed test cell successfully

Test cell used:

```python
print("Jupyter is connected to my environment")
import sys
print(sys.version)
print(sys.executable)
```

Result:

* Code executed without errors
* Python executable path confirms correct Conda environment

Conclusion:

* Jupyter launches correctly
* Kernel executes Python code
* Jupyter is connected to intended environment

---

## 5. Environment Summary

| Component   | Status   | Version    |
| ----------- | -------- | ---------- |
| Python      | Verified | 3.11.6     |
| Conda       | Verified | 23.7.4     |
| Environment | Active   | base       |
| Jupyter     | Verified | Functional |

---

## Final Confirmation

The local development environment has been fully verified.

* Python is accessible and stable
* Conda environments activate correctly
* Jupyter Notebook/Lab executes Python successfully
* All components are integrated and functional

This machine is certified and ready for the Data Science sprint.

<<<<<<< HEAD
=======
---

## Verification Artifacts

* Pull Request: (Add PR link here)
* Walkthrough Video: (Add video link here)

>>>>>>> 66e10b79853d3424c69b722713680e6eb3540227
---