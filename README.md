# Ensemble Stock Trading Bot - reinforcement learning 

Forked AI4Finance-LLC\FinRL to make and ensemble bot that trades stockes to learn more about reinforcement learning algorithms.

## Contributors

Jeff Keith

Ian MacPherson

Bretton Tan

Paul Messina

## Instalation

Please See our final report: Final Report - CPS824 Group 8.pdf and code/main.py.

To setup up environment follow the steps below:

1. Create new venv and run pip install -r requirements.txt
2. Replace Line 249 in finrl/env/env_stocktrading.py with:

```python
if len(self.df.tic.unique()) > 1:
	self.turbulence = self.data['turbulence'].values[0]
else:
	self.turbulence = self.data['turbulence']
```	


Once the venv has been set up, experiments can be run by changing the stock ticker on line 33 of code/main.py

Or models can be trained by adjusting the parameters starting at line 78 of code/main.py


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
