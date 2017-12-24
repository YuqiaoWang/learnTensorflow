import tensorflow as tf
import numpy as np
import xlrd

#添加层
def add_layer(inputs, in_size, out_size, activation_function=None):	
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases;
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    parameters = [outputs, Weights, biases]
    return parameters

# 1.训练的数据
# Make up some real data
#x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
#noise = np.random.normal(0, 0.05, x_data.shape)
#y_data = np.sin(x_data) + 0.5 + noise;
# 1.5 从excel中读取数据
workbook = xlrd.open_workbook(r'/home/yuqiaowang/learnTensorflow/data/sheet.xlsx')
sheet1 = workbook.sheet_by_index(0)
x_data_raw = []
y_data_raw = []
for i in range(1, sheet1.ncols):
	xdata = sheet1.col_values(i)[:31]
	ydata = sheet1.col_values(i)[31:]
	x_data_raw.append(xdata)
	y_data_raw.append(ydata)

x_data = np.array(x_data_raw)
y_data = np.array(y_data_raw)

#x_data = np.array(sheet1.col_values(0))[:, np.newaxis]
#y_data = np.array(sheet1.col_values(2))[:, np.newaxis]



# 2.定义节点准备接收数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 31])
ys = tf.placeholder(tf.float32, [None, 16])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs(31个神经元), 在隐藏层有30个神经元
layerparameters1 = add_layer(xs, 31, 40, activation_function=tf.nn.relu)
l1 = layerparameters1[0]
Weights1 = layerparameters1[1]
biases1 = layerparameters1[2]

# 隐藏层2
#layerparameters2 = add_layer(l1, 30, 20, activation_function = tf.nn.relu)
#l2 = layerparameters2[0]


# add output layer 输入值是隐藏层l1,在预测层输出1个结果
layerparameters3 = add_layer(l1, 40, 16, activation_function=None)
prediction = layerparameters3[0]
Weights3 = layerparameters3[1]
biases3 = layerparameters3[2]
# 4.定义loss表达式
# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

# 5.选择optimizer使loss达到最小
#这一行定义了用什么方式去减少loss,学习率是0.1
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)

# import step 对所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()
# 上面定义的都没有运算，直到sess.run 才会开始运算
sess.run(init)

# 迭代1000次学习， sess.run optimizer
for i in range(100000):
# training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 500 == 0:
        # to see the step improvement
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
#print(sess.run(tf.matmul(Weights1, Weights2)))
#print(sess.run(Weights1))
#print(sess.run(Weights2))
#print(sess.run(biases1))
#print(sess.run(biases2))
# 验证部分
#for i in range(800, 850):
#    x_valiation = x_data[i][:, np.newaxis]
#    hidden_valiation = np.add(np.matmul(sess.run(Weights1), x_valiation), sess.run(biases1))
#   y_valiation = np.add(np.matmul(sess.run(Weights2), hidden_valiation), sess.run(biases2))
#    print(sess.run(y_valiation))

#for i in range (800, 820):
#    print(sess.run(prediction, feed_dict={xs:x_data[i][:, np.newaxis]}))
print(sess.run(prediction, feed_dict={xs:x_data[5]}))
