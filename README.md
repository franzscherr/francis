# francis
Just a growing collection of useful functionality and imports to enhance prototyping speed in machine learning similar environments

It becomes useful when one uses IPython or Jupyter notebooks extensively

E.g.: Instead of
```Python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots(1, figsize=(16, 8))
ax.plot(x, y)
plt.show()
```
abbreviate imports and figure creation:
```Python
from francis import *

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

ax = oax()
ax.plot(x, y)
plt.show()
```
