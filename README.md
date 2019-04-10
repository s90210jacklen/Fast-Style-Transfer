# Fast Style Transfer 快速風格轉移 (Using Tensorflow)
由《Perceptual Losses for Real-Time Style Transfer and Super-Resolution》[1] 所提出，基於原始的Style Transfer利用預訓練好的VGGNet之外，需要再訓練一個影像產生網路(Image Tranform Net)，當訓練完成後，將圖片放入Image Tranform Net後只需計算一遍就能快速輸出具有風格的圖片


# Concept
![arch](https://github.com/s90210jacklen/Fast-Style-Transfer/blob/master/images/arch.png)

由上圖的架構圖可得知系統由兩個Neural Network所組成，左邊為影像生成網路(Image Tranform Net)，而右邊損失網路(Loss Network)就是VGG-16的架構

- **訓練階段** : 利用Loss Network來定義內容損失(Content Loss)與風格損失(Style Loss)，目標是讓影像生成網路(Image Tranform Net)輸入一張風格圖片後經由訓練能夠有效生成圖片
- **執行階段** : 輸入一張圖片，經由影像生成網路(Image Tranform Net)生成出一張風格轉移過後的圖片

# Detail　
與原始的Style Transfer的概念相同，利用損失網路(Loss Network)來定義內容損失(Content Loss)與風格損失(Style Loss)



- **內容損失(Content Loss)**</br>
使用VGG-16的relu3_3層輸出的特徵



- **風格損失(Style Loss)**</br>
使用VGG-16的relu1_2，relu2-2，relu3_3，relu4_3共四個層的特徵

- 同樣的利用內容損失與風格損失組合成一個總損失(Toatal Loss)，並利用總損失來訓練影像生成網路(Image Tranform Net)
![Total loss](https://github.com/s90210jacklen/Fast-Style-Transfer/blob/master/images/total_loss.png)

# Tricks
- 不採取常見的轉置卷積(Transposed Convolution)的方式，而是先放大再做卷積，這樣可以消除棋盤狀的noise</br>
由[此篇文章](https://distill.pub/2016/deconv-checkerboard/)所提出
```python
def resize_conv2d(x, input_depth, output_depth, ksize, strides, traning):
    # 先放大
    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value
        width = x.get_shape()[2].value
        
        new_height = height * strides * 2
        new_width = width * strides * 2
        
        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
     # 再卷積   
        return conv2d(x_resized, input_depth, output_depth, ksize, strides)
```
- 使用Instance Norm取代常見的Batch Norm</br>
由[《Instance Normalization: The Missing Ingredient for Fast Stylization》](https://arxiv.org/abs/1607.08022)所提出
```python
def instance_norm(x):
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))
```
結合上述方式定義deconv
```python
with tf.variable_scope('deconv1'):
        deconv1 = relu(instance_norm(resize_conv2d(res5, 128, 64, 3, 2, training)))
    with tf.variable_scope('deconv2'):
        deconv2 = relu(instance_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training)))
    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))
```
- 整個架構中有影像生成網路(Image Tranform Net)與損失網路(Loss Network)，而目標是訓練影像生成網絡，因此只需訓練與儲存影像生成網路的變數
```python
variable_to_train = []  
# 使用tf.trainable_variables()找出所有可以訓練的變數
for variable in tf.trainable_variables(): 
    # 如果不在損失網路中，把他們加入列表variables_to_train
    if not(variable.name.startswith(FLAGS.loss_model)):  
        variable_to_train.append(variable)  
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step, var_list=variable_to_train)  
```

# Usage

**On Windows**
```bash
eg: With Tensorflow as backend
> python eval.py --model_file folder_name/denoised_starry.ckpt-done --image_file img/ponbao.jpg 
```
folder_name : 指定一個存放模型檔案的資料夾名稱，將預訓練的模型檔案(此為denoised_starry.ckpt-done)放到此資料夾內</br>
img : 指定一個存放圖片的資料夾名稱，將圖片放到此資料夾內</br>
denoised_starry.ckpt-done : 以梵谷的星空最為風格影像的預訓練模型檔
