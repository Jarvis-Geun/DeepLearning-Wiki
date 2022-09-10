# Unpacking 사용법(asterisk)

### Reference
> https://yeko90.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%A4%91%EA%B8%89-unpacking%EC%97%90-%EB%8C%80%ED%95%B4%EC%84%9C-%EC%9E%98-%EC%95%8C%EA%B3%A0-%EA%B3%84%EC%8B%9C%EB%82%98%EC%9A%94-%EC%82%AC%EC%9A%A9%EB%B2%95

<br>

## 1. asterisk(*)을 사용한 Unpacking
### 1.1 code
```python
a, b, c = [1, 2, 3]
print(a)
print(b)
print(c)
```

```text
# 출력결과

1
2
3
```
- 출력결과를 보면 알 수 있듯이, 변수의 개수와 리스트 내의 값의 개수가 일치하므로 정상적으로 `unpacking`이 되는 것을 확인할 수 있다.

<br>

### 1.2 code
```python
d, e, f = [1, 2, 3, 4, 5]
print(d)
print(e)
print(f)
```
```text
# 출력결과

ValueError: too many values to unpack (expected 3)
```
- 이전 예시와 달리, 변수의 개수와 리스트 값의 개수가 일치하지 않으면 위와 같은 에러가 발생하는 것을 확인할 수 있다.

<br>

### 1.3 code
그렇다면 d, e, f에 어떻게든 강제로 리스트 내의 값을 할당하고 싶다면 어떻게 해야할까? 방법은 아래코드를 통해 확인할 수 있다.

```python
# asterisk(*)을 사용한 unpacking
d, *e, f = [1, 2, 3, 4, 5]
print(d)
print(e)
print(f)
```
```text
# 출력결과

1
[2, 3, 4]
5
```
- asterisk가 없는 변수에 우선 리스트 값이 할당되고, 남은 asterisk가 있는 변수(e)에 나머지 리스트 값들이 할당된 것을 확인할 수 있다.

<br>

### 1.4 code
위와 같은 방법으로 다른 변수에 asterisk를 할당해보았다.
```python
d, e, *f = [1, 2, 3, 4, 5]
print(d)
print(e)
print(f)
```
```text
# 출력결과

1
2
[3, 4, 5]
```
- 이번에도 마찬가지로 asterisk가 있는 변수(f)에 나머지 리스트 값들이 할당된 것을 확인할 수 있다.

<br>

### 1.5 code
문득 궁금한 점이 생겼다. asterisk가 있는 변수의 값이 리스트 형태로 출력되는 것을 확인할 수 있는데, 이는 곧 그 변수에 대하여 인덱싱이 가능할 것으로 생각되어 실험해보았다.

```python
d, *e, f = [1, 2, 3, 4, 5]
print("e[0] : ", e[0])
print("e[1] : ", e[1])
print("e[2] : ", e[2])
print("type(e)", type(e))
```
```text
e[0] :  2
e[1] :  3
e[2] :  4
type(e) <class 'list'>
```
- 예상과 동일하게 인덱싱이 가능한 것을 확인할 수 있다.

<br>

### 1.6 code
반면에 asterisk(*)을 두 개 이상의 변수에 사용할 경우 에러가 발생하는 것을 확인할 수 있다.

```python
d, *e, *f = [1, 2, 3, 4, 5]
```

```text
# 출력결과

SyntaxError: two starred expressions in assignment
```

<br>

## 2. Container 자체를 Unpacking 하는 asterisk(*)
### 2.1 code
```python
list = [1, 2, 3]
print("list : ", list)
print("*list : ", *list)

asterisk = []
print("asterisk : ", asterisk)

asterisk = [*list]
print("asterisk : ", asterisk)
```
```text
# 출력결과

list :  [1, 2, 3]
*list :  1 2 3
asterisk :  []
asterisk :  [1, 2, 3]
```

- asterisk를 사용하여 `*list`을 출력하면 `1 2 3`의 값이 출력되는 것을 확인할 수 있다.
- 이렇게 출력된 값을 임의의 리스트에 `[*list]`의 형태로 추가할 수 있다.

<br>

### 2.2 code : 두 개의 리스트 값을 하나의 리스트에 통합하기
위의 방법을 통해 두 개 이상의 리스트 내의 값을 하나의 리스트에 통합할 수 있다.
```python
list_1 = [1, 2, 3]
list_2 = [4, 5, 6]
print("list_1 : ", list_1)
print("list_2 : ", list_2)

integrated = [*list_1, *list_2]
print("integrated : ", integrated)
```

```text
list_1 :  [1, 2, 3]
list_2 :  [4, 5, 6]
integrated :  [1, 2, 3, 4, 5, 6]
```

<br>

### 2.3 code : tuple에 unpacking하기
```python
list_1 = [1, 2, 3]
print("list_1 : ", list)

list_2 = [4, 5, 6]
print("list_2 : ", list_2)

tuple = (*list, *list_2)
print("tuple : ", tuple)
```

```text
list_1 :  [1, 2, 3]
list_2 :  [4, 5, 6]
tuple :  (1, 2, 3, 4, 5, 6)
```
- 위와 같이, 두 개의 list를 한 개의 tuple로 통합할 수 있다.

<br>

### 2.3.1 code : 한 개의 list를 한 개의 tuple로 통합하기
asterisk를 사용하여 한 개의 리스트 값을 한 개의 tuple로 변환하려 시도했지만 불가능한듯 하다.

```python
list = [1, 2, 3]
print("list : ", list)

tuple = (*list)
print(tuple)
```

```text
# 출력결과

SyntaxError: can't use starred expression here
```
- 리스트의 경우, 한 개의 리스트에 대해서 asterisk 사용이 가능하였지만 tuple의 경우 그렇지 않은 것 같다.
`list = [list_1]`의 값과 `list = [*list_1]`의 값은 다르지만 tuple의 경우 그렇지 않다.

- `tuple = (*list)`는 불가능하며, `tuple = (list)`는 사용가능하다.

```python
list = [1, 2, 3]
print("list : ", list)

tuple = (list)
print("tuple : ", tuple)
```

```text
# 출력결과

list :  [1, 2, 3]
tuple :  [1, 2, 3]
```

<br>

### 2.3.2 code : tupe(list) 사용
참고로, 이전의 `tuple = (list)` 방법 이외에도 tuple 함수를 사용하여 list를 tuple로 변환할 수 있다.

```python
list = [1, 2, 3]
print("list : ", list)

list_to_tuple = tuple(list)
print("list_to_tuple : ", list_to_tuple)
```

```text
# 출력결과

list :  [1, 2, 3]
list_to_tuple :  (1, 2, 3)
```

<br>

## 3. 딕셔너리를 사용하여 두 개의 asterisk(**) 활용하기
### 3.1 code : 딕셔너리와 한 개의 asterisk(*) 활용

```python
dic_1 = {'a' : 1, 'b' : 2}
dic_2 = {'c' : 3, 'd' : 4}
dic_3 = {'e' : 5, 'a' : 6, 'f' : 7}

print("dic_1 : ", dic_1)
print("dic_2 : ", dic_2)
print("dic_3 : ", dic_3)
```
```text
dic_1 :  {'a': 1, 'b': 2}
dic_2 :  {'c': 3, 'd': 4}
dic_3 :  {'e': 5, 'a': 6, 'f': 7}
```

<br>

```python
dic_one_asterisk = {*dic_1, *dic_2, *dic_3}
print("dic_one_asterisk : ", dic_one_asterisk)
```

```text
# 출력결과

dic_one_asterisk :  {'f', 'e', 'd', 'a', 'c', 'b'}
```
- 한 개의 asterisk(*)을 딕셔너리와 함께 사용할 경우, 위와 같은 결과를 얻을 수 있다.
- `dic_one_asterisk`는 set(집합)의 형태로 정의되어있다. 따라서 무작위 순서로 값들이 배열되어있다.
- `key`만 출력되는 것을 확인할 수 있다.

<br>

### 3.2 code : 두 개의 asterisk(**) 사용하기
```python
dic_1 = {'a' : 1, 'b' : 2}
dic_2 = {'c' : 3, 'd' : 4}
dic_3 = {'e' : 5, 'a' : 6, 'f' : 7}

dic_two_asterisk = {**dic_1, **dic_2, **dic_3}
print("dic_two_asterisk : ", dic_two_asterisk)
```

```text
# 출력결과

dic_two_asterisk :  {'a': 6, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 7}
```
- key와 value 둘 다 출력되는 것을 확인할 수 있다.
- 딕셔너리를 순서대로 차곡차곡 저장한 것을 확인할 수 있다.
- 중복선언된 `'a'`의 경우, 가장 최근의 key와 value가 저장된 것을 확인할 수 있다. 딕셔너리에서 `key는 오직 한 개만 존재`할 수 있기 때문이다.

<br>

## 4. nested unpacking
`nested list` : 리스트 안에 리스트가 선언되어 있는 형태

```python
list_ex = [1, 2, [3, 4]]
a, b, c, d = list_ex
```

```text
# 출력결과

ValueError: not enough values to unpack (expected 4, got 3)
```
- 위와 같이 코드를 짜면 에러가 발생하는 것을 확인할 수 있다. 해결방법은 아래와 같다.

<br>

```python
list_ex = [1, 2, [3, 4]]
a, b, (c, d) = list_ex

print("a : ", a)
print("b : ", b)
print("c : ", c)
print("d : ", d)
```

```text
a :  1
b :  2
c :  3
d :  4
```

> * 참고사항  
위에서 변수를 list나 tuple로 정의한 경우가 있는데 되도록이면 `함수 이름과 겹치지 않게 변수를 선언`해주는 것을 추천한다. 코드를 실행하다보면 이따금씩 에러가 발생하는 경우도 있으며, 변수와 함수를 헷갈릴 수 있기 때문이다.