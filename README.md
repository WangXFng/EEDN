## EEDN (SIGIR'23) 
	The paper can be found in /paper for read-only. 
	DO NOT distribute it until it's published by ACM SIGIR around July 23th 2023.

### Run
	python Main.py

### Note
* Configures are given by Constants.py and Main.py
* As mentioned in paper, EEDN requires a shallow and wide architecture, please **DO NOT** over limit the embedding size for comparisons, unless there are not enough GPU memories.
* When you apply EEND on other datasets, as **$\lambda$** and **$\delta$** are sensitive, please tune these two hyperparameters by optuna at least 100 times, which **HAS BEEN IMPLEMENTED** by the given code in Main.py (Line.160)
* If you have any problem, please feel free to contact me by kaysenn@163.com.

### Dependencies
* Python 3.7.6
* [PyTorch](https://pytorch.org/) version 1.7.1.
___

### Datasets
	Three files are required: train.txt (for training), tune.txt (for tuning), and test.txt (for testing).
	Each line denotes a interaction including a user visited a POI at times.
	The format is [#USER_ID]\t[#POI_ID]\t[#TIMES]\n, which is the same for all files.
	For example,
	0	0	1
	0	1	3
	0	3	2
	1	2	1
	the user (ID=0) visited the POI (ID=0) at 1 times, 
				  the POI (ID=1) at 3 times, 
				  and the POI (ID=3) at 2 times.
	the user (ID=1) visited the POI (ID=2) at 1 times.


<table>
	<tr> <td> Dataset</td> <td> #Users</td> <td> #Items</td> <td> lambda</td> <td> delta </td> </tr>
	<tr> <td> Douban-book</td> <td> 12,859</td> <td> 22,294</td> <td> 0.5</td> <td> 1 </td> </tr>
	<tr> <td> Gowalla</td> <td> 18,737</td> <td> 32,510</td> <td> 1.5 </td> <td> 4 </td> </tr>
	<tr> <td> Foursquare</td> <td> 7,642</td> <td> 28,483</td> <td> 0.4</td> <td> 0.7 </td></tr>
	<tr> <td> Yelp challenge round 7</td> <td> 30,887</td> <td> 18,995</td> <td> 1</td> <td> 2.4 </td></tr>
	<tr> <td> Yelp2018</td> <td> 31,668</td> <td> 38,048</td> <td> 1</td> <td> 4 </td></tr>
</table>


### Baselines
* [SimGCL](https://github.com/Coder-Yu/QRec) SIGIR'2022
* [NCL](https://github.com/RUCAIBox/NCL) WWW'2022
* [DirectAU](https://github.com/THUwangcy/DirectAU) KDD'2022
* [STaTRL](https://github.com/WangXFng/STaTRL) APIN'2022
* [SGL](https://github.com/wujcan/SGL-TensorFlow) SIGIR'2021
* [SEPT](https://github.com/Coder-Yu/QRec) KDD'2021
* [LightGCN](https://github.com/gusye1234/LightGCN-PyTorch) SIGIR'2020
* [CPIR](https://repository.kaust.edu.sa/bitstream/handle/10754/667564/Conference%20Paperfile1.pdf?sequence=1) IJCAI'2020
* [ENMF](https://github.com/chenchongthu/ENMF) TOIS'2020
* [SAE-NAD](https://github.com/allenjack/SAE-NAD) CIKM'2018

### Citation
If this repository helps you, please cite:

	@inproceedings{wang2023eedn,
	  title={EEDN: Enhanced Encoder-Decoder Network with Local and Global Interactions for POI Recommendation},
	  author={Wang, Xinfeng and Fukumoto, Fumiyo and Cui, Jin and Suzuki, Yoshimi and Li, Jiyi and Yu, Dongjin},
	  booktitle={Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
	  year={2023}
	}

### Acknowledge
Thanks to [Coder-Yu](https://github.com/Coder-Yu/SELFRec) who collected many available baselines, and kindly released.
