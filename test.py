import numpy as np
import matplotlib.pyplot as plt

M = np.array([25, 30, 35, 40])

E_Q20 = np.array([4.1136e-1, 9.6433e-5, 9.1030e-5])
E_Q40 = np.array([9.2825e-1, 3.9613e-4, 3.9162e-6])

import numpy as np
import matplotlib.pyplot as plt

M = np.array([3, 5, 7,10])

E_Q20 =np.array([
    1.4475,
    1.8515e-4,
    6.8630e-8,
    1.0725e-9
])
E_Q40 = np.array([
    3.3169e-1,
    3.5595e-5,
    3.2090e-6,
    1.6407e-6
])

plt.figure()

plt.semilogy(M, E_Q20, 'o-', label='N=1')
plt.semilogy(M, E_Q40, 's-', label='N=2')

plt.xlabel('Number of features M')
plt.ylabel("Maximum error")
plt.title('Semi-log plot')
plt.grid(True, which='both')
plt.legend()

plt.show()