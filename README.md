## EEDN (SIGIR 2023)

### Run (Configures are given by Constants.py and Main.py)
> python Main.py


### Note
* As mentioned in paper, EEDN requires a shallow and wide architecture, please **DO NOT** over limit the embedding size for comparisons.
* When you apply EEND on other datasets, as **$\lambda$** and **$\delta$** are sensitive, please tune the these two hyperparameters by optuna at least 100 times, which **HAS BEEN IMPLEMENTED** by the given code in Main.py (line.162)
* If the memory of your GPU server is less than 24G, please small the embedding size in Main.py (line.145-148).
* If you have any problem, please contact me by kaysenn@163.com.


### Dependencies
* Python 3.7.6
* [Anaconda](https://www.anaconda.com/) 4.8.2 contains all the required packages.
* [PyTorch](https://pytorch.org/) version 1.7.1.


### Datasets
<table>
	<tr> <td> Dataset</td> <td> #Users</td> <td> #Items</td> </tr>
	<tr> <td> Douban-book</td> <td> 12,859</td> <td> 22,294</td> </tr>
	<tr> <td> Gowalla</td> <td> 18,737</td> <td> 32,510</td> </tr>
	<tr> <td> Foursquare</td> <td> 7,642</td> <td> 28,483</td> </tr>
	<tr> <td> Yelp challenge round 7</td> <td> 30,887</td> <td> 18,995</td> </tr>
	<tr> <td> Yelp2018</td> <td> 31,668</td> <td> 38,048</td> </tr>
</table>


### Acknowledge
Thanks to [Coder-Yu](https://github.com/Coder-Yu/SELFRec) who collected many available baselines.
