### 证明双线性插值中心重合

1. 原图宽度为width，高度为height， 中心坐标则为（（height-1）/2, （width-1）/2)
2. 扩缩图比例为ratio，则扩缩图宽度为width\*ratio， 高度为height\*ratio中心坐标为（(height\*ratio-1)/2, (width\*ratio-1)/2)
3. 双线性插值是要找最邻近的四个点
4. 所以上面扩缩图的中心坐标当成是dst像素点，要求它在原图上的坐标，如果要中心对称的话，必须正好等于原图的中心坐标。
5. 设dstX=(height\*ratio -1) /2 ，那么 (dstX + 0.5) / ratio  - 0.5= (height\*ratio/2 )/ratio -0.5 = (height-1)/2 = srcX,;
6. 同理设dstY=(width\*ratio -1) /2 ，那么 (dstY + 0.5) / ratio  - 0.5= (width\*ratio/2 )/ratio -0.5 = (width-1)/2 = srcY, 
7. 因此可以证明srcX = (dstX + 0.5) *scaleX -0.5这个公式可以让我们的中心重合