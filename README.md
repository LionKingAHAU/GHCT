
# 0. Prerequisite 

  

> If you want to use the GHCT model, you might need to be equipped with the env below：

  

|Package|Version|
|---|---|
|python|3.8.18|
|numpy|1.22.3|
|pandas|1.3.5|
|matplotlib|2.12.0|
|scikit-learn|1.3.0|
|torch|2.0.1|
|torch_geometric|2.4.0|
|networkx|3.1|


  

# 1. Brief Introduction 

  

we propose a novel approach called **G**raph residual connection attention **H**olistic processing multiple **C**hannels graph **T**ransformer(**GHCT**) for efficiently predicting interactions between lncRNAs and proteins. The core of GHCT lies in utilizing holistic attention processing. Specifically, we introduce an identity feature matrix to help the model better handle nodes within the graph, allowing the model to adaptively extract information from latent feature spaces, particularly in feature learning.

The figure illustrates the overall architecture of our model.

  

# 2. How to Use GHCT

 
> Our project is comprised of six main files.
> Please make sure that the **data** file and the **GHCT** file are in the same directory. 
> Then, **enter the GHCT directory and Run the commands as follows step by step**.


main.py: this script uses GHCT model to predict lncRNA-protein interactions.

```python

python main.py

```


# 3. Available Data


>Publicly available datasets were analyzed in this study. This data can be found here: Publicly available datasets were analyzed in this study. NPInter2.0 database can be found https://github.com/zhanglabNKU/ BiHo-GNN/tree/main/BiHo/dataset_preprocessing/dataset,NPInter3. 0 database can be found http://bigdata.ibp.ac.cn/npinter4/download/, RPI2241 database can be found https://github.com/zhanglabNKU/ BiHo-GNN/tree/main/BiHo/dataset_preprocessing/dataset.Fullcodes of the BiHo-GNN project are available at our GitHub repository https:// github.com/zhanglabNKU/BiHo-GNN.



> In the `dataset_preprocessing/dataset` dir, we provide the data of NPInter v2.0 database, NPInter v3.0_H and NPInter v3.0_M .


> In the `data` dir. To prevent information leakage and to fairly evaluate the model, we split the three lncRNA-protein interaction datasets into a 5:5 ratio, we provide the data used in our paper. 
  
# 4. Download Links of Five State-of-the-Art (SOTA) Methods for Comparison in paper

|Method  |Paper Link  | Code Link |
|--|--|--|
|BiHo-GNN  |[https://doi.org/10.3389/fgene.2023.1136672](https://doi.org/10.3389/fgene.2023.1136672)  | [https://github.com/zhanglabNKU/BiHo-GNN](https://github.com/zhanglabNKU/BiHo-GNN) |
|LPI-HyADBS  |[https://doi.org/10.1186/s12859-021-04485-x](https://doi.org/10.1186/s12859-021-04485-x)  | [https://github.com/plhhnu/LPI-HyADBS](https://github.com/plhhnu/LPI-HyADBS)  |
|LPIGAC  |[https://doi.org/10.1109/BIBM52615.2021.9669316](https://doi.org/10.1109/BIBM52615.2021.9669316)  | [https://github.com/zhanglabNKU/LPIGAC](https://github.com/zhanglabNKU/LPIGAC) |
|LncPNet  |[https://doi.org/10.3389/fgene.2021.814073](https://doi.org/10.3389/fgene.2021.814073)  | [https://github.com/zpliulab/LncPNet](https://github.com/zpliulab/LncPNet) |
|LPI-SKF |[https://doi.org/10.3389/fgene.2020.615144](https://doi.org/10.3389/fgene.2020.615144)  | [https://github.com/zyk2118216069/LPI-SKF](https://github.com/zyk2118216069/LPI-SKF) |
