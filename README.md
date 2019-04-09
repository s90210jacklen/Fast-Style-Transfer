# Fast Style Transfer 快速風格轉移 (Using Tensorflow)
由《Perceptual Losses for Real-Time Style Transfer and Super-Resolution》[1] 所提出，基於原始的Style Transfer利用預訓練好的VGGNet之外，需要再訓練一個影像產生網路(Image Tranform Net)，當訓練完成後，只須將圖片放入Image Tranform Net後只需計算一遍就能快速輸出具有風格的圖片


# Concept
![arch](https://github.com/s90210jacklen/Fast-Style-Transfer/blob/master/images/arch.png)

由上圖的架構圖可得知系統由兩個Neural Network所組成，左邊為影像生成網路(Image Tranform Net)，而右邊損失網路(Loss Network)就是VGG-16的架構

- **訓練階段** : 利用Loss Network來定義內容損失(Content Loss)與風格損失(Style Loss)，目標是讓影像生成網路(Image Tranform Net)輸入一張風格圖片後經由訓練能夠有效生成圖片
- **執行階段** : 輸入一張圖片，經由影像生成網路(Image Tranform Net)生成出一張風格轉移過後的圖片

## Detail
與原始的Style Transfer的概念相同，利用損失網路(Loss Network)來定義內容損失(Content Loss)與風格損失(Style Loss)

- 內容損失(Content Loss)
![content_loss]https://github.com/s90210jacklen/Fast-Style-Transfer/blob/master/images/content_loss.png
- 風格損失(Style Loss)
![style_loss]https://github.com/s90210jacklen/Fast-Style-Transfer/blob/master/images/style_loss.png
# Tricks




