# 2021.11.17(WED) 공부내용

## 1. numpy.unique() 함수


### 1.0 Reference
> https://www.delftstack.com/ko/api/numpy/python-numpy-unique/


### 1.1 예시 코드
```python
# np.unique() 사용법
import numpy as np
a = np.array([[2, 3, 4],
              [5, 4, 7],
              [4, 2, 3]])

np.unique(a)
```
```
# 출력 결과

array([2, 3, 4, 5, 7])
```


### 1.2 return_index=True 사용
```python
print(a, end='\n\n')

np.unique(a, return_index=True)
```
```
# 출력 결과

[[2 3 4]
 [5 4 7]
 [4 2 3]]

(array([2, 3, 4, 5, 7]), array([0, 1, 2, 3, 5]))
```
- 주어진 평탄화 된 입력 배열에서 정렬 된 고유 값 배열의 튜플과 각 고유 값의 `첫 번째 발생 인덱스 배열`을 제공합니다.


### 1.3 return_counts=True 사용
```python
print(a, end='\n\n')

np.unique(a, return_counts=True)
```
```
# 출력 결과

[[2 3 4]
 [5 4 7]
 [4 2 3]]

(array([2, 3, 4, 5, 7]), array([2, 2, 3, 1, 1]))
```
- 주어진 평면화 된 입력 배열에서 정렬 된 고유 값 배열의 튜플과 입력 배열에 `각 고유 값의 개수` 배열을 제공합니다.


### 1.4 return_inverse=True
```python
print(a, end='\n\n')

np.unique(a, return_inverse=True)
```
```
# 출력 결과

[[2 3 4]
 [5 4 7]
 [4 2 3]]

(array([2, 3, 4, 5, 7]), array([0, 1, 2, 3, 2, 4, 2, 0, 1]))
```
- 주어진 평면화 된 입력 배열에서 정렬 된 고유 값 배열의 튜플과 고유 배열의 인덱스 배열을 제공합니다.
- 여기서 2는 평면화 된 배열의 첫 번째 위치와 두 번째 마지막 위치에서 발생합니다. 마찬가지로 어떤 위치에서 어떤 값이 발생하는지 찾을 수 있습니다.


### 1.5 axis 매개 변수 사용
```python
b = np.array([[2,3,2],
            [2,3,2],
           [4,2,3]])

print(b, end='\n\n')

print(np.unique(b, axis=0), end='\n\n')
print(np.unique(b, axis=1), end='\n\n')
print(np.unique(b, axis=-1), end='\n\n')
```
```
# 출력 결과

[[2 3 2]
 [2 3 2]
 [4 2 3]]

[[2 3 2]
 [4 2 3]]

[[2 2 3]
 [2 2 3]
 [3 4 2]]

[[2 2 3]
 [2 2 3]
 [3 4 2]]
```