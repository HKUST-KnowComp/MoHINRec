# MoHINRec
The code CIKM 19 paper "[Motif Enhanced Recommendation over Heterogeneous Information Network]

Readers are welcomed to fork this repository to reproduce the experiments and follow our work. Please kindly cite our paper

    @inproceedings{zhao2019mohinrec,
    title={Motif Enhanced Recommendation over Heterogeneous Information Network},
    author={Zhao, Huan and Zhou, Yingqi and Song, Yangqiu and Lee, Dik Lun},
    booktitle={CIKM},
    year={2019}
    }
    
We use Epinions Dataset and Ciao Dataset from https://www.cse.msu.edu/~tangjili/trust.html. Any problems, you can create an issue. Note that these two datasets are provied by Prof. [Jiliang Tang](https://www.cse.msu.edu/~tangjili/trust.html), thus if you use these datasets for your paper, please cite the authors' paper as instructed in the website https://www.cse.msu.edu/~tangjili/trust.html 

## Instructions

For the sake of ease, a quick instruction is given for readers to reproduce the whole process on Epinions dataset. Note that the programs are testd on **Linux(CentOS release 6.9), Python 2.7 and Numpy 1.14.0 from Anaconda 4.3.6.**

### Prerequisites

1. Create a directory "data" in this project directory, download epinions dataset and put it under "data/" .
2. Create directory **"log"** in the project by "mkdir log".
3. Create directory **"fm\_res"** in the project by "mkdir fm\_res".
4. Open preprocess_E.py in this project directory, set the value of "dir_" equals "data/epinions/", and then run 
```python
python preprocess_E.py
```
5. Iteratively create directories **"sim_res/path_count"** and **"mf_features/path_count"** in directory **"data/epinions/exp_split/1/"**.
### Meta-graph Similarity Matrices Computation.
To generate the MoHINRec M1-M7 similarity matrices with alpha from 0 to 1 on Epinions dataset, run

	python e_commu_mat_computation.py epinions 1
The arguments are explained in the following:
	
	epinions: specify the dataset.
	1: run for the split dataset 1, i.e., exp_split/1
This command generates MoHINRec M1-M7 similarity matrices with alpha from 0 to 1. One dependent lib is bottleneck, you may install it with "**pip install bottleneck**".

### Meta-graph Latent Features Generation.
To generate the latent features by MF based on the simiarity matrices, run
    
    python mf_features_generator.py epinions 1

This command generates the latent features for MoHINRec M1-M7 similarity matrices. The arguments are the same as the above ones.

Note that, to improve the computation efficiency, some modules are implements with C and called in python(see *load_lib* method in mf.py). Thus to successfully run mf\_features\_generator.py, you need to compile two C source files. The following scripts are tested on CentOS, and readers may take as references.

	gcc -fPIC --shared setVal.c -o setVal.so
	gcc -fPIC --shared partXY.c -o partXY.so

After the compiling, you will get two files in the project directory "setVal.so" and "partXY.so".

### FMG
After obtain the latent features, then the readers can run FMG model as following:
    
    python run_exp.py config/epinions.yaml -reg 0.5

One may read the comment in files in directory config for more information.

## Misc
If you have any questions about this project, **you can open issues**, thus it can help more people who are interested in this project.
I will reply to your issues as soon as possible.
