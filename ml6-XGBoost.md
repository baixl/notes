# 机器学习系列6:XGBoost
>date: 2018-05-21
>categories: 机器学习 
>tags: xgboost
>mathjax: true
>---
> 该系列将整理机器学习相关知识。这篇博客主要讨论:
> 1 XGBoost的算法原理
> 2 XGboost参数调优
> 
> 由于本人水平有限，目前也是在持续学习中，这篇博客在[原论文](http://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)和[XGBoost官网](https://xgboost.readthedocs.io/en/latest/)基础上整理，如有出入请留言说明，非常感谢

XGBoost是**Extreme Gradient Boosting**的简称，Gradient Boosting是论文"Greedy Function Approximation: A Gradient Boosting Machine"中介绍的梯度提升算法。Boosting Tree树数据挖掘和机器学习中常用算法之一，其对输入要求不敏感，效果好，在工业界用的较多(kaggle比赛必用)。
<!--more-->
## 1 背景知识
### 1.1 Traing loss + Regularization
XGBoost用于监督学习问题（分类和回归）。监督学习的常用目标函数是：
通常目标函数包含两部分:训练误差和正则化
$$obj(θ)=L(θ)+Ω(θ)$$
其中L是损失函数,度量模型预测与真实值的误差。常用的损失函数：
预测问题的平方损失函数：
$$L(\theta) = \sum_{i}{(y_i-\hat{y_i})}^2 $$
logistic 损失：
$$L(\theta) = \sum_{i}[y_i ln(1+e^{-\hat{y_i}}) +(1-y_i) ln(1+e^{\hat{y_i}})] $$
$Ω$是正则化项，度量模型的复杂度，避免过拟合，常用的正则化有L1 和L2正则化。
### 1.2 Tree模型融合（集成）
Boosting Tree 最基本的部分是**回归树**(GBDT中用到的也是回归树，而不是分类树)，也即是CART（如下图），CART会把输入根据属性分配到各个叶子节点上，而每个叶子节点上面会对应一个分数值。下面的例子是预测一个人是否喜欢电脑游戏。将叶子节点表示为分数之后，可以做很多事情，比如概率预测，排序等等。
![](http://7xnzwk.com1.z0.glb.clouddn.com/15149016431943.jpg)
一个CART往往过于简单，而无法有效的进行预测，因此更加高效的是使用多个CART进行融合，使用集成的方法提升预测效率：
![](http://7xnzwk.com1.z0.glb.clouddn.com/15147876733214.jpg)

假设有两颗回归树，则两棵树融合后的预测结果如上图。用公式表示为：
$$\hat{y_i}=\sum_{k=1}^{K}f_k(x_i),f_k\in\mathscr{F}$$
其中， K是树的个数， $f_k(x_i)$是第k棵树对于输入 $x_i$ 输出的得分， $f_k $是相应函数， $\mathscr{F}$ 是相应函数空间。则目标函数为：
$$obj(\theta)=\sum_i^n L(y_i,\hat{y_i})+\sum_{k=1}^K\Omega(f_k)$$

函数$L$描述$y_i$ ， $\hat{y_i}$之间的距离。
常用的模型融合方法是Random Foreast和Boosting Tree,这两种方法构造树的方式不同(*参考系列前面的集成学习一节*)。Tree Ensemble中，模型的参数是什么呢？其实就是指树的结构和叶子节点上面分数的预测。如何求参数？定义目标函数，通过优化目标函数进行求解。
### 1.3 Tree Boosting
假设每次迭代生成一棵树，则训练目标函数可以写成：
$$obj(\theta)^{t}=\sum_i^n l(y_i,\hat{y_i}^{(t)})+ \sum_{k=1}^t\Omega(f_k)$$

其中第一部分是训练误差，第2部分是每棵树的复杂度。$\hat{y_i}^{(t)}$为第t步迭代的预测值，且有以下迭代关系：
$\hat{y_i}^{(0)}=0 $
$\hat{y_i}^{(1)}=f_1(x_i) = \hat{y_i}^{(0)}+f_1(x_i)$
$\cdots$
$\hat{y_i}^{(t)}= \sum_{k=1}^t(f_k(x_i)= \hat{y_i}^{(t-1)}+f_t(x_i)$

则训练目标函数可以写成：
$$obj^{(t)} =\sum_{i=1}^nl(y_i,\hat{y_i}^{(t)})+\sum_{i=1}^t\Omega(f_i) $$
$$obj^{(t)} =\sum_{i=1}^nl(y_i,\hat{y_i}^{(t-1)}+ f_t(x_i))+\Omega(f_t)+  \sum_{i=1}^{t-1}\Omega(f_i) $$

对于第t步来说$\sum_{i=1}^{t-1}\Omega(f_i)$是已知的，因此有:
$$obj^{(t)} =\sum_{i=1}^nl(y_i,\hat{y_i}^{(t-1)}+ f_t(x_i))+\Omega(f_t)+ constant $$

如果$l$使用**平方损失函数**，则有

$obj^{(t)} =\sum_{i=1}^n [ y_i-(\hat{y_i}^{(t-1)}+ f_t(x_i)]^2+\Omega(f_t)+constant$

$=\sum_{i=1}^n[y_i^2-2y_i(\hat{y_i}^{(t-1)}+ f_t(x_i))+(\hat{y_i}^{(t-1)}+ f_t(x_i))^2]+\Omega(f_t)+  constant$

$=\sum_{i=1}^n[y_i^2-2y_i\hat{y_i}^{(t-1)}- 2y_if_t(x_i)+ (\hat{y_i}^{(t-1)})^2+2\hat{y_i}^{(t-1)}f_t(x_i) +(f_t(x_i))^2]+\Omega(f_t)+  constant$

$=\sum_{i=1}^n[2{(\hat{y_i}^{(t-1)} - y_i)}f_t{(x_i)} +(f_t(x_i))^2]+  \sum_{i=1}^n[{(y_i)}^2-2y_i\hat{y_i} + {(\hat{y_i}^{(t-1)})}^2 ]+\Omega(f_t)+ constant$

其中对于第t步来说，$\sum_{i=1}^n[{(y_i)}^2-2y_i\hat{y_i} + {(\hat{y_i}^{(t-1)})}^2 ]$也是常数，所以 目标函数优化为：
$$obj^{(t)}=\Sigma_{i=1}^n[2{(\hat{y_i}^{(t-1)} - y_i)}f_t{(x_i)} +(f_t(x_i))^2]+\Omega(f_t)+ constant$$

其中$(\hat{y_i}^{(t-1)} - y_i)$一般叫做残差。

当使用平方损失函数时，拟合残差的步骤就是上次分享中的Adaboost算法。更加一般地，对于不是平方损失函数（当对于一般损失函数时，可以用一阶梯度拟合残差，对应的就是GBDT方法。
### 1.4 GBDT回顾(参考上一节的GBDT)
提升树利用加法模型和前向分步算法实现学习的优化过程。当损失函数时平方损失和指数损失函数时，每一步的优化很简单，如平方损失函数学习残差回归树。
当为一般的损失函数时，GBDT利用最速下降的近似方法，即利用损失函数的负梯度在当前模型的值，作为回归问题中提升树算法的残差的近似值，拟合一个回归树。

提升树利用加法模型和前向分步算法实现学习的优化过程。当损失函数时平方损失和指数损失函数时，每一步的优化很简单，如平方损失函数学习残差回归树。
当为一般的损失函数时，GBDT利用最速下降的近似方法，即利用损失函数的负梯度在当前模型的值，作为回归问题中提升树算法的残差的近似值，拟合一个回归树。

##  2 XGBoost
### 2.1 目标函数
更加一般地,对于一般损失函数，XGBoost会使用泰勒展开的形式进而用到二阶导数。目标函数：
$$obj^{(t)} =\sum_{i=1}^nl(y_i,\hat{y_i}^{(t-1)}+ f_t(x_i)+\Omega(f_t)+ constant $$
用泰勒展开来近似目标函数：
（在GBDT中，使用梯度，只是使用了一阶导数的形式） 

* 泰勒展开： $f(x+ \Delta x)\approx f(x) + f^{'}(x)\Delta x + \frac{1}{2} f^{''}  \Delta x^2$
* 定义：
$$g_i = \partial_{ \hat{y}^{(t-1)}} l(y_i,\hat{y}^{(t-1)})$$
$$h_i = \partial_{ \hat{y}^{(t-1)}}^2 l(y_i,\hat{y}^{(t-1)})$$
则有：
$$obj^{(t)} = \sum_{i=1}^n [l(y_i,\hat{y}^{(t-1)}+g_i f_t(x_i) +\frac{1}{2} h_if_t^2(x_i)] + \Omega(f_t)+ constant$$

移除常量：$l(y_i,\hat{y_i}^{(t-1)})$,则目标函数为：
$$obj^{(t)} = \sum_{i=1}^n [g_i f_t(x_i) +\frac{1}{2} h_if_t^2(x_i)] + \Omega(f_t)$$

有了这个更新后的目标函数后，可以看出这个目标函数仅仅依赖一阶似然的一阶和二阶导数。对于Adaboost，通常只能用用平方损失函数和通过拟合残差的方式来求解。GBDT在ADAboost基础上做了一部优化，用释然函数在每一个树处的负梯度来近似残差。
有了上述通用的形式后，就可以使用任意（可以求1阶和2阶）形式的损失函数。

### 2.2 树的复杂度
当目前为止，讨论了模型中训练误差的部分。下面来探讨模型复杂度$\Omega(f_t)$的表示方式。
重新定义每棵树，将树`f`拆分成树结构`q`和叶子权重部分`w`两部分。结构函数`q`把输入映射到叶子的索引上。而`w`给定了每个索引号对应叶子的分数。
![](http://7xnzwk.com1.z0.glb.clouddn.com/15147940969973.jpg)
当给定了如上图的树定义后，每颗树的复杂度可以定义为下面公式，这个公式里定义了树中叶子节点的个数和每颗树叶子节点的输出分数的L2正则项。
![](http://7xnzwk.com1.z0.glb.clouddn.com/15147943145691.jpg)
![](http://7xnzwk.com1.z0.glb.clouddn.com/15147943314091.jpg)
在这种定义下，我们可以把目标函数改写为：
![](http://7xnzwk.com1.z0.glb.clouddn.com/15147944731145.jpg)

其中，I被定义为每个叶子上面样本的集合：$I_j= \left( i|q(x_i)=j \right)$。这个目标函数包含了T个独立的单变量二次函数。

上述目标函数对$w_j$求导并令导数为0，可以求得：
$$w_j* = -\frac{G_j}{H_j+\lambda}$$
$$Obj = -\frac{1}{2}\sum_{j=1}^T \frac{G_j^2}{H_j+\lambda } + \gamma T$$

**计算举例：**
Obj代表了当我们指定一个树的结构的时候，我们在目标上面最多减少多少。我们可以把它叫做结构分数(structure score)。你可以认为这个就是类似吉尼系数一样更加一般的对于树结构进行打分的函数。下面是一个具体的打分函数计算的例子
![](http://7xnzwk.com1.z0.glb.clouddn.com/15148066903621.jpg)

所以目标很明确，不断的枚举树结构，然后利用上述打分函数来寻找一个最优结构的树，加入到我们的模型中，再重复这个操作。枚举树结构计算消耗太大，常用的是**贪心法**，每一次尝试对已经的叶子加入一个分割，对一个具体的分割方案，我们可以获得分割后的增益为：
$$Gain = \frac{1}{2} \left[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma$$

**如果Gain<0,则此节点不应该split成左右两支。**

对于每次扩展，我们还是要枚举所有可能的分割方案，如何高效地枚举所有的分割呢？我假设我们要枚举所有 `x<ax<a` 这样的条件，对于某个特定的分割aa我们要计算左边和右边的导数和。
实际应用中，先将 $g_i$ 从小到大排序，然后进行遍历，看每个结点是否需要split。
![](http://7xnzwk.com1.z0.glb.clouddn.com/15148069572544.jpg)
我们可以发现对于所有的aa，我们只要做一遍从左到右的扫描就可以枚举出所有分割的梯度和GL和GR。然后用上面的公式计算每个分割方案的分数就可以了。

### 2.3 其他优化方法

* shrinkage and Column Subsampling 

除了在目标函数中添加正则化外，还有两种防止过拟合的方法：Shrinkage and Column Subsampling
Shrinkage ： 在每一步生成boosting树时添加一个删减参数：η ，通SGD中的学习率类似，这个参数可以减少单颗树的作用
Column（Feature） Subsampling：这个技术在Random Foreast构建子树时使用（RF决策树的每个节点，随机在属性集合中选择k个属性子集用做划分属性）。XGBoost在构建子树时也引入了相似的优化方法。

* SPLIT FINDING ALGORITHM


### 2.4 对比GBDT
 （1）[xgboost相比传统gbdt有何不同？xgboost为什么快？xgboost如何支持并行？](http://wepon.me/2016/05/07/XGBoost%E6%B5%85%E5%85%A5%E6%B5%85%E5%87%BA/)

* 传统GBDT以CART作为基分类器，xgboost还支持线性分类器，这个时候xgboost相当于带L1和L2正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。
* 传统GBDT在优化时只用到一阶导数信息，xgboost则对代价函数进行了二阶泰勒展开，同时用到了一阶和二阶导数。顺便提一下，xgboost工具支持自定义代价函数，只要函数可一阶和二阶求导。
* xgboost在代价函数里加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、每个叶子节点上输出的score的L2模的平方和。从Bias-variance tradeoff角度来讲，正则项降低了模型的variance，使学习出来的模型更加简单，防止过拟合，这也是xgboost优于传统GBDT的一个特性。
* Shrinkage（缩减），相当于学习速率（xgboost中的eta）。xgboost在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。实际应用中，一般把eta设置得小一点，然后迭代次数设置得大一点。（补充：传统GBDT的实现也有学习速率）
* 列抽样（column subsampling）。xgboost借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算，这也是xgboost异于传统gbdt的一个特性。
* 对缺失值的处理。对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向。XGBoost对于确实值能预先学习一个默认的分裂方向
* xgboost工具支持并行。boosting不是一种串行的结构吗?怎么并行的？注意xgboost的并行不是tree粒度的并行，xgboost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。xgboost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。
* 可并行的近似直方图算法。树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以xgboost还提出了一种可并行的近似直方图算法，用于高效地生成候选的分割点
（2）[机器学习算法中GBDT和XGBOOST的区别有哪些？](https://www.zhihu.com/question/41354392/answer/98658997)


## 3 参考文献
* [XGBoost: A Scalable Tree Boosting System](http://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
* [https://xgboost.readthedocs.io/en/latest/model.html](https://xgboost.readthedocs.io/en/latest/model.html)
* [论文笔记：XGBoost: A Scalable Tree Boosting System](https://zhuanlan.zhihu.com/p/30738432)
* [XGBoost 与 Boosted Tree](http://www.52cs.org/?p=429)
* [xgboost入门与实战（原理篇）](http://blog.csdn.net/sb19931201/article/details/52557382)
* [史上最详细XGBoost实战](https://zhuanlan.zhihu.com/p/31182879)




