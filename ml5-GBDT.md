# 机器学习系列5:GBDT
> 该系列将整理机器学习相关知识。这篇博客主要讨论:
> 1 GBDT
>  GBDT、Xgboost、LightGBM在机器学习中应用是否广泛，也是各种机器学习竞赛(kaggle)的常用方法。比如我在做kaggle的一些练习时，通常会用随机森林、GBDT作为基线方法，然后会再使用Xgboost、LightGBM做进一步优化，最后将上面几个方法的结果做个最总的融合（stacking）。

## 1 GBDT原理
原始论文地址：[Greedy Function Approximation:A Gradient Bossting MAchine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)

>原文很长，我这里只是简单介绍了GBDT的原理部分。最开始解除这个是在李航《统计学习方法》中，但李航书中介绍比较简单，这里参考一个博客，[GBDT小结](https://blog.csdn.net/niuniuyuh/article/details/76922210)，讲的比较好(下面部分内容来自该博客，感谢博主，引用请指名原出处)。

<!--more-->
提升树利用加法模型和前向分步算法实现学习的优化过程。当损失函数时平方损失和指数损失函数时，每一步的优化很简单，如平方损失函数学习残差回归树。

GBDT(Gradient Boosting Decision Tree)又叫GBRT(Gradient Boosting Regression Tree)、MRRT（Multiple Additive Regression Tree）。**GBDT中用到的树都是回归树，不是分类树**，GBDT也使用迭代决策算法，每一轮迭代生成的树都是拟合上一轮的残差，但和一般提升树不同的是，GBDT会用损失函数的负梯度（残差的减少方向）来拟合残差，叫伪残差。

假设我们的优化目标是：
$$F^*(x) = arg\min_{F(x)}E_{y,x}[L(y,F(x))]$$
预测函数$F(x)$的参数为：$P={ P_1,P_2...P_M } $，$ P_m={\beta_m,\alpha_m }$，第m个弱分类器或者回归树表示为：$\beta_mh(x;\alpha_m)$,$\beta_m$ 为弱分类器的系数(权重)，$\alpha_m$表示其参数，则有：

那么对于样本$[ {x_i,y_i } ]^N$,优化问题变为：
$$(\beta_m,\alpha_m)= arg\min_{\alpha,\beta}(\sum_{i=1}^N) L(y_i,F_{m-1}(x_i)+\beta h(x_i,\alpha))$$
GBDT的一般迭代过程为：

（1）初始化弱分类器器为常数$\rho$, $F_0(x) =arg \min_{\rho} \sum_{i=1}^N L(y_i,\rho)$
（2）在每次迭代中基于**回归树**构建弱分类器，假设第m次迭代后得到的预测函数为$F_m(x)$，相应的损失函数为$L(y,F_m(x))$,为使损失函数减少最快，第m个弱分类器$\beta_m h(x;\alpha_m)$应建立在前m-1次迭代生成的预测损失函数的负梯度(下降)方向。 用$-g_m(x_i)$表示第m次迭代的弱分类器的建立方向，$L(y_i, F(x))$表示前m-1次迭代生成的预测损失函数,表示为$L(y_i,F(x_i)) = (y_i-F(x_i))^2$，则有：

$$-g_m(x_i)=-[\frac{\partial{L(y_i,F(x_i))}}{\partial F(x_i)}]_{F(x_i)=F_{m-1}(x_i)}$$
上式就求出了梯度下降的方向，基于此可以求出参数$\alpha_m$和步长$\beta_m$:
$$\alpha_m = arg\min_{\alpha, \beta} \sum_{i=1}^N[-g_m(x_i)- \beta h(x_i, \alpha)]^2 $$
$$\beta_m = arg\min_{\beta} \sum_{i=1}^N L(y_i,F_{m-1}(x_i)+ \beta h(x_i;\alpha_m)) $$

（3）更新预测函数：$F_m(x) = F_{m-1}(x) + \beta_m h(x, \alpha_m)$,当相应的损失函数满足收敛条件或者生成预定的M个时，终止迭代。

## 2 参考文献
* [Greedy Function Approximation:A Gradient Bossting MAchine](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf)
* [GBDT的小结（来自论文greedy function approximation: a gradient boosting machine)](https://blog.csdn.net/niuniuyuh/article/details/76922210)
* 李航《统计学习方法》 


