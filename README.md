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
- 不採取常見的轉置卷積(Transposed Convolution)的方式，而是先放大再做卷積，這樣可以消除棋盤狀的noise
由[此篇文章](https://distill.pub/2016/deconv-checkerboard/)所提出
```python
with tf.variable_scope('deconv1'):
        deconv1 = relu(instance_norm(resize_conv2d(res5, 128, 64, 3, 2, training)))
    with tf.variable_scope('deconv2'):
        deconv2 = relu(instance_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training)))
    with tf.variable_scope('deconv3'):
        # 到這裡生成的圖片大小已經和原圖相同，所以不再進行反卷積
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))
```



