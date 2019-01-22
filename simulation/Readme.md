# How to use

+ Open the scene first in UI (test.ttt file) and code in  

```lua {.line-numbers}
simRemoteApi.start(19999)  
-- which is the port number in your connection  
```  

+ Start the simulation in UI.  
+ Run your own py!

## About Camera.py

Env: **Python 3.6 with Numpy&PIL / Ubuntu16.04**
Save depth & rgb image in the path

```python {.line-numbers}
Save_PATH_COLOR = r'./color/'  
Save_PATH_DEPTH = r'./depth/'  
```

The simulation steps is Simu_STEP  

```python {.line-numbers}
Simu_STEP = 10
```