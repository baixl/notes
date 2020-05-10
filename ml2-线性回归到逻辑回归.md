# 线性回归&逻辑回归
> 在广义线性模型中，线性回归假设数据服从正太分布，LR服从伯努利分布(0-1分布)

## 1、线性回归
线性回归的目的是用一条曲线拟合数据，常用于在房价、股票预测等。这里将从误差的概率分布解释线性回归。假设线性回归的预测值和真实值表示为：
$$y^i=\theta^T x^i + \varepsilon^i $$

在线性回归中，假设误差$\varepsilon^i$是`独立同分布`（误差有大有小，并且服从正太分布），并且服从均值为0，方差为$\theta^2$的高斯分布,则有：
$$ p(\varepsilon^i) = \frac{1}{\sqrt{2\pi\delta} } exp(-\frac{(\varepsilon^i)^2}{2\delta^2})$$
合并上面两个式子，则有
$$ p(y^i|x^i;\theta) = \frac{1}{\sqrt{2\pi\delta} } exp(-\frac{(y^i-\theta^Tx^i)^2}{2\delta^2})$$
假设样本集合大小为m，可以用极大似然估计对参数$\theta$进行估计，在线性回归中似然函数可以表示为：
$$ L(\theta) = \prod_{i=1}^{m}( p(y^i|x^i;\theta) =\prod_{i=1}^{m} \frac{1}{\sqrt{2\pi\delta} } exp(-\frac{(y^i-\theta^Tx^i)^2}{2\delta^2})$$
对数似然函数（将乘法转换为加法）：
$$ l(\theta) = \sum_{i=1}^{m} \log \frac{1}{\sqrt{2\pi\delta} } exp(-\frac{(y^i-\theta^Tx^i)^2}{2\delta^2}) $$
$$ =m \log( \frac{1}{\sqrt{2\pi\delta}}) - \frac{1}{2\delta^2 } \sum_{i=1}^{m} (y^i-\theta^Tx^i)^2$$
由上式子可以得到目标函数（求极小值）：
$$J(\theta) = \frac{1}{2}\sum_{i=1}^{m} (y^i-\theta^Tx^i)^2$$
这个式子就是**线性回归的最小二乘法**。
## 2、logistic regression
### 2.1、模型假设
逻辑斯谛回归是一种用于**分类**的机器学习算法，虽然叫回归，但实际上是用来做分类的，其本质上是一种线性模型。其假设模型是：
$$h_{\theta}(x) = g(\theta ^T x)$$
其中，$g(z) = \frac{1}{1+e^{-z}}$,则有：
$$h_{\theta}(x) = \frac{1}{1+e^{-\theta ^T x}}$$
其中，$h_{\theta}(x)$表示分类为正样本的概率，即$h_{\theta}(x) = p(y=1 | x;\theta)$。以上只是得出了样本点是正例的概率，到底预测它是正例还是负例，我们还需要一个`decision boundary`，例如：
$$ h_{\theta}(x) \ge 0.5 \rightarrow y=1 \ h_{\theta}(x) \lt 0.5 \rightarrow y=0 $$
由逻辑斯谛函数的性质，可得到：
$$\theta ^T x \ge 0 \rightarrow y=1 ,  \theta ^T x \lt 0 \rightarrow y=0$$

### 2.2、极大似然估计

二元分类可以看成一个伯努利分布，即0-1分布，上面提到
$$p(y=1 | x;\theta)=h_{\theta}(x) $$
$$p(y=0 | x;\theta)= 1- h_{\theta}(x) $$
合并上述式子则有：
$$p(y | x;\theta) = (h_{\theta}(x))^y (1 - h_{\theta}(x^i))^{(1-y)}$$
由极大释然估计，假设所有样本独立同分部，将它们的概率相乘得到似然函数：
$$ L(\theta) = \prod_{i=1}^{m}((h_{\theta}(x^i))^{y^i} (1 - h_{\theta}(x^i))^{(1-y^i)}) $$
取对数得到对数似然函数:
$$  l(\theta) = \sum_{i=1}^{m} (y^i \log(h_{\theta}(x^i)) + (1-y^i) (1 - h_{\theta}(x^i)) )$$

### 2.3、逻辑回归的损失函数
逻辑斯谛回归的损失函数定义为：
$$
J(\theta) = \frac{1}{m}\sum_{i=1}^{m} (Cost(h_{\theta}(x^i)), y^i)
$$
其中：
$$
Cost(h_{\theta}(x), y) = -\log(h_{\theta}(x)) \ \ if \ \ y=1 \ ;Cost (h_{\theta}(x), y) = -\log(1 - h_{\theta}(x)) \ \ if \ \ y=0
$$
结合两个式子，则简写为：
$$
Cost(h_{\theta}(x), y) = - y \log(h_{\theta}(x)) - (1-y) \log(1 - h_{\theta}(x))
$$
用图像表示为：
![](http://7xnzwk.com1.z0.glb.clouddn.com/15259312368518.jpg)
可以直观的看到，就是概率预测的和标签越接近惩罚越小，反之越大。当然，这里讲的只是二元分类，标签不是0就是1.

最后，逻辑回归的损失函数为

$$
J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} (y^i \log(h_{\theta}(x^i)) + (1-y^i) \log (1-h_{\theta}(x^i))
$$
### 2.4、梯度下降优化过程
使用梯度下降进行优化：
$$\theta_j = \theta_j - \alpha \frac{\partial J(\theta)}{\partial \theta_j}$$

其中：

$$J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} ((y^i \log(g(z)) + (1-y^i) \log (1-g(z)) )$$

$$\frac{\partial J(\theta)}{\partial \theta_j} = - \frac{1}{m} \sum_{i=1}^{m} ( \frac{y^i}{g(z)} \frac{\partial g(z)}{\theta_j} - \frac{1-y^i}{1-g(z)} \frac{\partial g(z)}{\theta_j})$$

$$
= - \frac{1}{m} \sum_{i=1}^{m}   \frac{y^i - g(z)}{g(z)(1-g(z))}  \frac{\partial g(z)}{\theta_j}
$$
$$\frac{\partial g(z)}{\theta_j} = g(z)(1-g(z)x_j^i$$
带入上式即可都得到
$$
\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)x_j^i
$$
这就是逻辑斯谛回归的梯度下降更新式子：
$$
\theta_j = \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_{\theta}(x^i)-y^i)x_j^i
 $$
## 3 广义线性模型GLM
讨论广义线性模型之前，先问下为`什么逻辑斯谛回归中要使用sigmod(x)函数`？
本质上，线性回归和逻辑回归都是广义线性模型的特例。具体推导参考知乎文章[广义线性模型](https://zhuanlan.zhihu.com/p/22876460)和[为什么 LR 模型要使用 sigmoid 函数，背后的数学原理是什么？](https://www.zhihu.com/question/35322351)
## 4 参考
* [logistic regression的公式手推相关](http://frankchen.xyz/2017/01/09/logistic-regression/)
* [美团点评:Logistic Regression 模型简介](https://tech.meituan.com/intro_to_logistic_regression.html)
* [逻辑回归常见面试总结](http://www.cnblogs.com/ModifyRong/p/7739955.html)
* [广义线性模型](https://zhuanlan.zhihu.com/p/22876460)


