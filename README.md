
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

  
The figure illustrates the overall architecture of our model.
![image](https://github.com/LionKingAHAU/GHCT/blob/main/The%20framework%20of%20GHCT.png)

# 2. How to Use GHCT

 
> Our project is comprised of six main files.
> Please make sure that the **data** file and the **GHCT** file are in the same directory. 
> Then, **enter the GHCT directory and Run the commands as follows step by step**.


main.py: this script uses GHCT model to predict lncRNA-protein interactions.

```python

python main.py

```
If you want to use multiprocessing, you can run the following example.
```python

python main.py --use_multiprocessing True

```

 
# 3. Available Data


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
