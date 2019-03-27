# machine_learning
机器学习、数据挖掘小练习存档。

- Iris_classification、face_recognition_and_similar_face_found、handwriting_recognition、newsgroups_classification是研究生期间“机器学习”课程作业。这几份作业为了熟悉流程，所以所有的输入处理都是使用下载的文件，而不是直接调用load()获取数据。


## 1、Iris_classification
该练习主要基于鸢尾花数据集，首先对数据分布进行了分析，接着分别使用logistic回归、决策树和神经网络（MLP分类器）对数据进行了训练和预测，并使用precision、recall和f1-score衡量模型预测的效果。

详细文档见：https://github.com/Nana0606/machine_learning/blob/master/Iris_classification/doc/experiment_report.pdf

（1）鸢尾花的数据分布如下：

<div align=center><img src="https://github.com/Nana0606/machine_learning/blob/master/imgs/iris_distribution.png" width="50%" alt="Iris数据分布"/></div>

由于数据具有4个维度，因为我们两两观察属性间的分布，从上图可以看出两两属性之间的分界线还是比较明显的。

（2）结果比较

<div align=center><img src="https://github.com/Nana0606/machine_learning/blob/master/imgs/iris_result_compare.png" width="50%" alt="Iris结果比较"/></div>

观察上图可知，Logsitic回归和MLP分类器对算法的测试结果都是完全正确的，其precision、recall和f1-score都是1。因此，就此数据集而言，决策树的性能更差一些。

## 2、face_recognition_and_similar_face_found
该练习是分成两部分，第一部分是使用机器学习进行人脸分类识别，并给出识别准确率，第二个任务是使用聚类或分类算法发现表情相似的脸图。对于第一个任务，本实验使用卷积神经网络（CNN）进行分类，这个任务相对简单，准确率也比较高，在测试集上达到了100%。对于第二个任务，尝试使用了Multinomial Naive Bayes（MNB）、Random Forest（RF）和CNN进行分类，准确率都不是非常高。使用MNB并使用五折交叉验证得到的评价准确率是28.59%；使用CNN准确率基本在25%左右；使用RF得到的准确率是41%。

详细文档见：https://github.com/Nana0606/machine_learning/blob/master/face_recognition_and_similar_face_found/doc/experiment_report.pdf

（1）人脸分类识别结果

<div align=center><img src="https://github.com/Nana0606/machine_learning/blob/master/imgs/face_recognition_result.png" width="50%" alt="人脸分类识别结果"/></div>

在原始使用Adam作为优化器时，准确率为99.8%左右，而后将Adam换成了RMSprop，算法的准确率达到了100%，实验设置epoches=10，batch_size=16。当优化器为RMSprop时，训练集上每个epoches后的损失值和准确率变化如上图。

（2）相似人脸发现

<div align=center><img src="https://github.com/Nana0606/machine_learning/blob/master/imgs/similar_face_found_result.png" width="50%" alt="相似人脸发现"/></div>

实验对比结果如图，使用Multinomial Naive Bayes结果可以达到28.59%，使用Random Forest结果可以达到41.0%，使用CNN可以达到28.9%，使用CNN并使用图片增强后的准确率可以达到32%。对于表情相似度发现的任务，因为脸部表情属于面部微表情，可能只是嘴角做出上扬的动作，表情就发生了变化，因此其分类比较困难，准确率比较低。另外，训练数据也比较少。

## 3、handwriting_recognition

该练习是使用神经网络进行MNIST手写体数字的识别。在神经网络的设计过程中，使用了单隐层神经网络，并使用relu作为激活函数，这也是目前使用最广泛、普遍效果最好的激活函数。除此之外，为了提高算法的性能，本实验使用分别使用正则化、滑动平均模型和指数衰减学习率对算法进行了优化，在MNIST手写体数据集上精度取得了98.42%的效果。为了对比随机梯度下降（SGD）的效果，将其与Adam进行了对比。因MNIST数据集过于简单，所以在本文中将此网络运用到Fashion-MNIST数据集中，这个数据集的识别难度比MNIST大，已经逐渐在替代MNIST数据集，在Fashion-MNIST数据集上，算法效果比在MNIST数据集上差一些。

详细文档见：https://github.com/Nana0606/machine_learning/blob/master/handwriting_recognition/doc/experiment_report.pdf

（1）测试集精度和验证集精度变化

<div align=center><img src="https://github.com/Nana0606/machine_learning/blob/master/imgs/mnist_train_validate.png" width="50%" alt="测试集精度和验证集精度变化"/></div>

从图中可以看出，对于MNIST数据集，验证集上的精度和测试集上的精度变化趋势是基本保持一致的，说明这个神经网络可以用于手写体的识别。另外，神经网络在MNIST数据集上的收敛速度比较快，若在比较复杂的数据集上训练，收敛速度应该会慢一些。

（2）不同模型精度对比

<div align=center><img src="https://github.com/Nana0606/machine_learning/blob/master/imgs/mnist_different_models_accuracy.png" width="50%" alt="不同模型精度对比"/></div>

从图中可以看出，调整神经网络的结构对最终的正确率有非常大的影响。在没有隐藏层时的精度只能达到92.7%，不用激活函数的状态下精度只能达到92.66%。而且不用滑动平均、正则化和指数衰减学习率时的精度和使用所有优化情况下的差别不是很大。这说明神经网络的结构对最终模型的效果有着本质的影响。所以在使用深度学习解决实际任务时，要根据任务的复杂程度设计网络层数，并通过不断调参，确定最优层数。另外，使用Adam优化方法的精度达到了98.45%，比使用SGD算法的精度稍高，说明使用Adam优化算法的效果更好一些。

（3）MNIST和Fashion-MNIST上精度对比

<div align=center><img src="https://github.com/Nana0606/machine_learning/blob/master/imgs/compare_mnist_fashion.png" width="50%" alt="MNIST和Fashion-MNIST上精度对比"/></div>

最优情况下，MNIST数据集在学习率是0.8时取得最好的效果，在测试集上的精度为98.42%；Fashion-MNIST数据集在学习率是0.6时取得最好的效果，在测试机上的精度是89.22%，因此同样的网络下，算法在较复杂的数据集上的效果要差些。

## 4、newsgroups_classification
该练习是对新闻文本进行分类，并使用五折交叉验证结果。本次实验使用的是卷积神经网络（CNN），并使用了glove词向量作为预处理词向量，使用了三层CNN构建模型，最终模型的accuracy达到了96.4%。除此之外，使用了朴素贝叶斯分类器作为对比方法。实验证明，CNN的效果优于使用朴素贝叶斯分类器的方法。

详细文档见：https://github.com/Nana0606/machine_learning/blob/master/newsgroups_classification/doc/experiment_report.pdf

（1）训练集和测试集效果对比

<div align=center><img src="https://github.com/Nana0606/machine_learning/blob/master/imgs/news_group_test_train_compare.png" width="50%" alt="训练集和测试集效果对比"/></div>

从结果可知，在训练集准确率达到了96.4%，在测试集上达到了94.2%左右。

（2）CNN和NB效果对比

<div align=center><img src="https://github.com/Nana0606/machine_learning/blob/master/imgs/news_group_cnn_nb.png" width="50%" alt="CNN和NB效果对比"/></div>

使用CNN模型和使用NB模型的执行结果对比如下图，从图中可以看出，相比较于NB，CNN具有绝对的优势。但是就执行效率来说，CNN效率较低，需要消耗大量的硬件资源，NB的效率更高些。

## pagerank
此文件夹下是pagerank实现代码，对应的详细博客地址为：https://blog.csdn.net/quiet_girl/article/details/81227904
- pagerank_impl.py是pagerank实现
- pagerank_nx.py是调用networkX库求解naive的pagerank和改进版的pagerank
