# click-fraud-cat-rfe
本项目是论文《CAT-RFE：一种点击欺诈的集成检测框架》的实验代码。

## 目录

- [项目目录](#project)
- [使用方法](#get-start)
- [项目声明](#statement)

<h2 id="project">项目目录</h2>

├─ raw_data (数据集目录)<br>
&emsp;├─ train.csv (训练集)<br>
&emsp;├─ test.csv (测试集)<br>
├─ initData.py (数据预处理)<br>
├─ baseline_model.py (基线模型)<br>
├─ voting_model.py (基于voting模型的RFE框架)<br>
├─ train_baseline.py (基线模型训练文件)<br>
├─ train_raw_ml.py (机器学习训练文件)<br>
├─ train_cat_rfe.py (RFE框架训练文件)<br>
├─ test.py (生成测试结果)<br>
├─ requirements.txt (项目依赖)<br>

> 以上列出了模型文件及主要的训练代码文件，其余未列出的文件均为项目基础文件，无需重点关注。<br>
> 本项目使用的数据集是百度飞桨的数据集，原链接如下：<br>
> https://aistudio.baidu.com/aistudio/competition/detail/52

<h2 id="get-start">使用方法 Getting Started</h2>

首先，拉取本项目到本地。<br>
First, pull the project to the local.

    $ git clone git@github.com:lyx199504/click-fraud-cat-rfe.git

接着，进入到项目中并安装本项目的依赖。但要注意，pytorch可能需要采取其他方式安装，安装完毕pytorch后可直接用如下代码安装其他依赖。<br>

    $ cd click-fraud-cat-rfe/
    $ pip install -r requirements.txt

然后，执行initData.py进行数据预处理。

最后，执行train_*.py等文件即可训练相应模型。

<h2 id="statement">项目声明</h2>

本实验代码基于param-opt训练工具，原项目作者及出处如下：<br>
**作者: Yixiang Lu**<br>
**项目: [param-opt](https://github.com/lyx199504/param-opt)**

若要引用本论文，可采用如下引用格式：<br>

    待定...

