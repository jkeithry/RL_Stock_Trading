# Reinforcement Learning Stock Trading Project
#### CPS 824 Group 8:
###### Jeff Keith
###### Ian MacPherson
###### Bretton Tan
###### Paul Messina

## Setup:

Please See our final report: Final Report - CPS824 Group 8.pdf and code/main.py.

To setup up environment follow the steps below:

1. Create new venv and run pip install -r requirements.txt
2. Replace Line 249 in finrl/env/env_stocktrading.py with:

>if len(self.df.tic.unique()) > 1:
>	self.turbulence = self.data['turbulence'].values[0]
>else:
>	self.turbulence = self.data['turbulence']
	
===============

Once the venv has been set up, experiments can be run by changing the stock ticket on line 33 of code/main.py

Or models can be trained by adjusting the parameters starting at line 78 of code/main.py