# SIN segmentation

### Reference
- [Recognition of Slab Identification Numbers Using a Fully
Convolutional Network](https://www.jstage.jst.go.jp/article/isijinternational/advpub/0/advpub_ISIJINT-2017-695/_pdf)

## 1. 모듈 불러오기 & 경로 지정

### 1.1 import module

```python
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
```

### 1.2 샘플 이미지 경로 지정(SIN 이미지)

```python
root_path = 'C:/Users/Geun/Google Drive/(2021) 3학년 2학기/MIL/MIL assignment/SIN/'
SIN_dir = root_path
train_dir = os.path.join(SIN_dir, 'Training set/Training')

train_fns = os.listdir(train_dir)
```

```python
# 샘플 이미지 경로 지정
sample_image_SIN_fp = os.path.join(train_dir, train_fns[0])
print("type(sample_image_SIN_fp) :", type(sample_image_SIN_fp))

sample_image_SIN = Image.open(sample_image_SIN_fp).convert('RGB')
print("type(sample_image_SIN) :", type(sample_image_SIN))
print("np.shape(sample_image_SIN) :", np.shape(sample_image_SIN))

plt.imshow(sample_image_SIN)
plt.show()
```

```
# 출력결과

type(sample_image_SIN_fp) : <class 'str'>
type(sample_image_SIN) : <class 'PIL.Image.Image'>
np.shape(sample_image_SIN) : (600, 960, 3)
```
![SIN_sample_image](img/2021.11.28(SUN)_1.png)


```python
sample_image_SIN = Image.open(sample_image_SIN_fp).convert('RGB')
print("np.shape(sample_image_SIN) :", np.shape(sample_image_SIN))
print(np.unique(sample_image_SIN))
```
```
# 출력결과

np.shape(sample_image_SIN) : (600, 960, 3)
[  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197
 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215
 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233
 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251
 252 253 254 255]
```



### 1.2 샘플 이미지 경로 지정(GTD 이미지)
```python
# GTD 이미지 경로 지정
GTD_dir = os.path.join(SIN_dir, 'Training set/GTD')
GTD_fp = os.listdir(GTD_dir)

sample_image_GTD_fp = os.path.join(GTD_dir, GTD_fp[0])
print(sample_image_GTD_fp)
```

```
# 출력결과

C:/Users/Geun/Google Drive/(2021) 3학년 2학기/MIL/MIL assignment/SIN/Training set/GTD\B17626571_1_PLACE1_SLABNUM1_TYPE0.png
```

```python
sample_image_GTD = Image.open(sample_image_GTD_fp).convert("RGB")
print("type(sample_image_GTD) :", type(sample_image_GTD))
print("np.shape(sample_image_GTD) :", np.shape(sample_image_GTD))

print("np.unique(sample_image_GTD) : ", np.unique(sample_image_GTD))

plt.imshow(sample_image_GTD)
plt.show()
```

```
# 출력결과

type(sample_image_GTD) : <class 'PIL.Image.Image'>
np.shape(sample_image_GTD) : (600, 960, 3)
np.unique(sample_image_GTD) :  [  0  20  40 100 120 140 220]
```
![GTD_sample_image](img/2021.11.28(SUN)_2.png)


---


## 2. np.resize(sample_image_SIN, (256, 256))

### 2.1 SIN 이미지 np.resize()
```python
sample_image_resize = np.resize(sample_image_SIN, (256, 256))
print("np.shape(sample_image_resize) :", np.shape(sample_image_resize))
print("np.unique(sample_image_resize) :", np.unique(sample_image_resize))
```

```
# 출력결과

np.shape(sample_image_resize) : (256, 256)
np.unique(sample_image_resize) : [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17
  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53
  54  55  56  57  58  59  60  61  62  63  64  65  66  67  68  69  70  71
  72  73  74  75  76  77  78  79  80  81  82  83  84  85  86  87  88  89
  90  91  92  93  94  95  96  97  98  99 100 101 102 103 104 105 106 107
 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125
 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143
 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179
 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197
 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215
 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233
 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251
 252 253 254 255]
```

- 1.2의 np.unique 값과 동일한 값을 가지는 것을 확인할 수 있다. 하지만 unique 값이 동일하다고 해서 이미지가 잘 resize됐다고 볼 수는 없다. resize된 이미지를 아래에 출력했는데 이미지가 잘 출력되지 않은듯 하다. 따라서 GTD image에 대해서 한번 더 resize를 해보고 np.unique 출력결과를 확인해보도록 한다.

```python
plt.imshow(sample_image_resize)
plt.show()
```

![](img/2021.11.28(SUN)_3.png)


### 2.2 GTD 이미지 np.resize()
```python
sample_image_GTD = Image.open(sample_image_GTD_fp).convert("RGB")
sample_image_GTD_resize = np.resize(sample_image_GTD, (256, 256))
print("np.shape(sample_image_GTD_resize) :", np.shape(sample_image_GTD_resize))
print("np.unique(sample_image_GTD_resize) : ", np.unique(sample_image_GTD_resize))

plt.imshow(sample_image_GTD_resize)
plt.show()
```

```
# 출력결과

np.shape(sample_image_GTD_resize) : (256, 256)
np.unique(sample_image_GTD_resize) :  [0]
```

![](img/2021.11.28(SUN)_4.png)

- 위의 결과를 통해 np.resize를 사용하면 이미지가 제대로 resize되지 않은 것을 확인할 수 있다. 이에 대한 자세한 내용은 [링크](https://okjh.tistory.com/140#but-reshape%EC%9D%80-%EC%9B%90%EC%86%8C%EC%9D%98-%EA%B0%9C%EC%88%98%EA%B0%80-%EB%8B%A4%EB%A5%B4%EB%A9%B4-%EC%98%A4%EB%A5%98%EA%B0%80-%EB%82%98%EC%A7%80%EB%A7%8C-resize%EB%8A%94-%EC%9E%90%EB%8F%99%EC%9C%BC%EB%A1%9C-%EB%A7%9E%EC%B6%B0%EC%A4%8C)를 참고한다.


---


## 3. transforms을 활용한 transforms.Resize()

이전 내용에서 확인할 수 있듯이, np.resize() 함수를 사용할 경우 원하는 사이즈로 resize가 되지않는 것을 확인할 수 있다. 따라서 다른 방법인 transforms.Resize() 함수를 사용해본다.

```python
sample_image_SIN = Image.open(sample_image_SIN_fp).convert("RGB")
sample_image_SIN = np.array(sample_image_SIN)

# transform 함수 정의
def transform(image) :
    transforms_ops = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize([256, 256]),
                        transforms.Normalize(mean = (0.485, 0.56, 0.406), std = (0.229, 0.224, 0.225))
    ])
    return transforms_ops(image)

# transform 함수 사용하여 Resize, Normalize
sample_SIN_transform = transform(sample_image_SIN)

plt.imshow(sample_SIN_transform)
plt.show()
```

```
# 출력결과

TypeError: Invalid shape (3, 256, 256) for image data
```
![](img/2021.11.28(SUN)_5.png)

- 위의 에러를 해결하기 위해 차원을 조정해주어야 한다. 자세한 내용은 [링크](https://stackoverflow.com/questions/65324466/typeerror-invalid-shape-3-32-32-for-image-data-showing-a-colored-image-in)를 참고한다.

<br>

### 3.1 transform을 활용하여 resize한 SIN sample image 출력
```python
sample_image_SIN = Image.open(sample_image_SIN_fp).convert("RGB")
sample_image_SIN = np.array(sample_image_SIN)

def transform(image) :
    transforms_ops = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize([256, 256]),
                        transforms.Normalize(mean = (0.485, 0.56, 0.406), std = (0.229, 0.224, 0.225))
    ])
    return transforms_ops(image)

sample_SIN_transform = transform(sample_image_SIN)

sample_SIN_transform = sample_SIN_transform.swapaxes(0, 2)
sample_SIN_transform = sample_SIN_transform.swapaxes(0, 1)

plt.imshow(sample_SIN_transform)
plt.show()
```

![](img/2021.11.28(SUN)_6.png)

- transforms.Normalize()를 사용할 경우, 이미지 원본이 변하는 것을 확인할 수 있다. `일단은 Normalize를 사용하지 않고 학습을 시켜본다. 이후에 Normalize를 사용해보도록 한다.`

<br>

### 3.2 transform을 활용하여 resize한 SIN sample image & GTD sample image 출력 (Normalize 사용 X)
```python
def transform(image) :
    transforms_ops = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize([256, 256]),
                        # transforms.Normalize(mean = (0.485, 0.56, 0.406), std = (0.229, 0.224, 0.225))
    ])
    return transforms_ops(image)


# SIN sample image
sample_image_SIN = Image.open(sample_image_SIN_fp).convert("RGB")
sample_image_SIN = np.array(sample_image_SIN)

# GTD sample image
sample_image_GTD = Image.open(sample_image_GTD_fp).convert("RGB")
sample_image_GTD = np.array(sample_image_GTD)

# transform SIN & GTD sample image
sample_SIN_transform = transform(sample_image_SIN)
sample_GTD_transform = transform(sample_image_GTD)

# 차원을 조정해주지 않으면 에러가 발생하므로 swapaxes 함수를 사용하여 차원을 조정해준다.
sample_SIN_swap = sample_SIN_transform.swapaxes(0, 2)
sample_SIN_swap = sample_SIN_swap.swapaxes(0, 1)
sample_GTD_swap = sample_GTD_transform.swapaxes(0, 2)
sample_GTD_swap = sample_GTD_swap.swapaxes(0, 1)

# image 출력
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].imshow(sample_SIN_swap)
axes[1].imshow(sample_GTD_swap)
plt.show()
```

![](img/2021.11.28(SUN)_7.png)

- 위의 코드는 커스텀 데이터셋을 정의할 때 수정하였다. 에러가 발생하고 np.unique로 값을 출력하였을 때, 원하는 값이 출력되지 않아 데이터셋의 정의할 때 수정하였다.


---


## 4. Define Dataset
- 이전의 코드는 데이터셋을 정의하고 나서 값을 확인할 때 에러가 발생하여 약간 수정하였다.
- 또한, GTD 이미지에 대하여 총 12개의 class에 대해 분류한 후, label_class 이미지로 지정해주었다. 즉, `KMeans(n_clusters = num_classes)`을 사용하여 GTD 이미지를 12개의 class로 나눈 후에 이를 SIN 이미지와 함께 학습하였다.

<br>

```python
class SINDataset(Dataset) :
    def __init__(self, SIN_dir, GTD_dir, label_model) :
        self.SIN_dir = SIN_dir
        self.GTD_dir = GTD_dir
        self.SIN_fns = os.listdir(SIN_dir)
        self.GTD_fns = os.listdir(GTD_dir)
        self.label_model = label_model
    
    def __len__(self) :
        return len(self.SIN_fns)

    # len는 하나의 값만 출력할 수 있기 때문에 SIN과 GTD의 길이 둘 다를 출력할 수는 없다.
    # 하지만 어차피 두 개의 길이는 같으므로 하나만 출력해도 상관없다.
    # def __len__(self) :
    #     return len(self.GTD_fns)

    def __getitem__(self, index) :
        SIN_fn = self.SIN_fns[index]
        GTD_fn = self.GTD_fns[index]
        
        SIN_fp = os.path.join(self.SIN_dir, SIN_fn)
        GTD_fp = os.path.join(self.GTD_dir, GTD_fn)
        
        SIN_image = Image.open(SIN_fp).convert("RGB")
        GTD_image = Image.open(GTD_fp).convert("RGB")
        
        SIN = np.array(SIN_image)
        GTD = np.array(GTD_image)

        GTD = self.label_model.predict(GTD.reshape(-1, 3)).reshape(600, 960)
        
        SIN = self.transform(SIN)
        label_class = self.transform(GTD).long()
        label_class = label_class.squeeze(dim=0)

        return SIN, label_class

    def transform(self, image) :
        transforms_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
            # transforms.Normalize(mean = (0.485, 0.56, 0.406), std = (0.229, 0.224, 0.225))
        ])
        return transforms_ops(image)
```

- 위의 코드에서 `label_class = label_class.squeeze(dim=0)`에 주목한다. 이를 추가하지 않을 경우, `SIN.shape`과 `label_class.shape`이 아래와 같다.

```
torch.Size([3, 256, 256])
torch.Size([1, 256, 256])
```

- squeeze(dim=0)을 추가할 경우 아래와 같은 결과를 얻을 수 있다.

```
torch.Size([3, 256, 256])
torch.Size([256, 256])
```

- 위의 값이 정확할지는 모델을 학습시켜봐야 결과를 알 수 있을 것으로 생각한다.  
`학습을 시킨 결과, 제대로 된 예측값을 얻었다. 따라서 label_class.squeeze(dim=0)은 이상이 없는 것으로 생각된다.`



### 4.1 label_model(KMeans) 정의
```python
num_items = 1000

# 0~255 숫자를 3*num_items번 랜덤하게 뽑기
color_array = np.random.choice(range(256), 3*num_items).reshape(-1,3)
print(color_array.shape)

# label_model 정의 (KMeans 사용)
num_classes = 12
label_model = KMeans(n_clusters = num_classes)
label_model.fit(color_array)
```

```
# 출력결과

(1000, 3)
KMeans(n_clusters=12)
```


### 4.2 정의된 데이터셋 점검
```python
dataset = SINDataset(train_dir, Ground_Truth_dir, label_model)
print(len(dataset))
```

```
# 출력결과

1844
```

<br>

```python
SIN, label_class = dataset[0]
print(SIN.shape)
print(label_class.shape)
```

```
# 출력결과

torch.Size([3, 256, 256])
torch.Size([256, 256])
```

<br>

```python
print(np.unique(SIN))
print(np.unique(label_class))
```

```
# 출력결과

[0.         0.00226716 0.00241268 ... 0.999977   0.9999924  1.        ]
[4 5 6 7 8]
```


---

## 5. Training the model

- model : U-Net
- criterion : nn.CrossEntropyLoss()
- optimizer : optim.Adam()
- learning rate : 0.01
- batch_size : 4
- epochs : 10

<br>

위의 내용을 바탕으로 코드를 구성하여 학습시켰다. 다만, 코드가 길어 첨부하지는 않고 [링크]()로 남긴다.

<br>

### 5.1 Results for training

```python
fig, axes = plt.subplots(1, 2, figsize = (10, 5))
axes[0].plot(step_losses)
axes[1].plot(epoch_losses)

plt.show()
```

![](img/2021.11.28(SUN)_8.png)

- 왼쪽 이미지 : `step_lossses`
- 오른쪽 이미지 : `epoch_losses`


---


## 6. Check model predictions

### 6.1 Define the Test Dataset
```python
class testDataset(Dataset) :
    def __init__(self, test_dir) :
        self.test_dir = test_dir
        self.test_fns = os.listdir(test_dir)
    

    def __len__(self) :
        return len(self.test_fns)


    def __getitem__(self, index) :
        test_fn = self.test_fns[index]
        
        test_fp = os.path.join(self.test_dir, test_fn)
        
        test = Image.open(test_fp).convert("RGB")
        
        test_image_array = np.array(test)
        
        test_image = self.transform(test_image_array)

        return test_image

    def transform(self, image) :
        transforms_ops = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([256, 256]),
            # transforms.Normalize(mean = (0.485, 0.56, 0.406), std = (0.229, 0.224, 0.225))
        ])
        return transforms_ops(image)
```


### 6.2 Visualize the predicted label class

![](img/2021.11.28(SUN)_9.png)

- epochs = 10으로 하여 학습했기 때문에 약간의 오차가 있는 것을 확인할 수 있다. 하지만 SIN 이미지에 대하여 label class가 거의 유사하게 예측이 된 것을 관찰할 수 있다.

- epochs = 20으로 하여 학습시키면 더 좋은 예측값을 얻을 수 있을 것으로 생각한다.

- `이미지가 흐리게 보이는데, 이유는 아직 못찾았다. 추후에 찾아서 고치도록 하자.`


---

\+ 추가내용

- Test 하는 과정에서 inverse_transform을 추가하였는데, 이 때문에 이미지가 흐릿하게 나왔다. 이를 제거하고 다시 test를 해보니, 선명하게 이미지가 출력되는 것을 확인할 수 있었다.

![](img/2021.12.02(THU)_1.png)


- 참고사항
```python
# 아래의 값들을 이용하여 학습시킨 결과가 위의 이미지이다.
batch_size  = 4

epochs = 20

lr = 0.001
```