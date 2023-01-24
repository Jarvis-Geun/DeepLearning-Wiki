# XAI (Explanable AI)
`-` Reference

- [An Overview of Model-Agnostic Interpretation Methods](http://dmqm.korea.ac.kr/activity/seminar/297)

<br>

## Model-Agnostic Methods
- `model-agnostic` : 모델에 구애받지 않고 독립적으로 모델을 해석할 수 있다는 의미

### Partial Dependence Plot (PDP)
`-` 장점

- 해석이 직관적이고 명확함
- 구현하기 쉬움 (관심 feature에 대해 marginalize 하기만 하면 됨)

`-` 단점

- 한 plot에 그려서 사람이 직관적으로 해석할 수 있는 feature 개수는 2개 (2차원)
- 모든 feature space에 대해 연산해야하므로 계산량이 많음
- 각 특성(feature)이 독립이라고 가정
- marginalize를 하면 곧 기댓값을 취하므로 각 n개의 점들이 갖는 분포가 무시됨

---

### Individual Conditional Expectation (ICE)
- PDP의 단점 개선
- Individual : 각 관측치에 dependence를 개별적으로 시각화
- 관심변수가 변할 때 어떻게 예측값(target)이 변하는지 모든 train 데이터에 대해 보여줌
- PDP는 ICE의 각 line들의 평균선임

<br>

`-` 장점

- 해석이 직관적이고 명확함
- PDP처럼 기댓값을 취하지 않기에, 각 관측치에 대응되는 선을 그릴 수 있음

`-` 단점

- 관측치 수가 많을 경우, 너무 조밀하게 plot 되어 제대로 파악하기 어려울 수 있음
- 그 외 단점은 PDP와 유사함

---

### Permutation Feature Importance
- Permutation : 확인하고자 하는 특성치(j열) 순서만을 shuffle 하여 새 데이터 행렬을 만들어 base 성능과의 차이를 `featrue importance`로 사용

---

### LIME (Local Interpretable Model-agnostic Explanations)
- SHAP의 기본이 되는 이론
- Local : 단일 관측치(혹은 데이터셋 일부분)에 대한 모델 예측값 해석에 대해 초점을 둠
- Surrogate model : 원래 모델 자체로 해석하기 어려울 때 외부에 구조가 간단한 대리(surrogate) 모델로 두어 해석 ➡️ Ex. linear regression model

<br>

`-` 아이디어

- 복잡한 데이터에 적합된 복잡한 모델의 전역적인 해석(global interpretation)은 어렵다.
- 국소적(local)으로는 비교적 해석이 간단한 모델(surrogate model)로 근사시킬 수 있다고 가정하면, 국소적인 해석(local interpretation)으로 설명할 수 있다. (Ex. surrogate model : Linear Regression)
- `Super-pixel`을 input으로 사용하여 사람이 해석할 수 있는 형태로 변환(segmentation)하여 surrogate model의 입력으로 사용함

<br>

`-` 장점/의의

- Global한 해석이 아닌 개별 데이터 인스턴스에 대한 local 해석력 제공
- Perturbation(작은 변화)의 방식을 다르게 하면 model-agnostic 하게 해석할 수 있는 도구를 제공함
- SHAP보다 적은 계산량

`-` 단점

- 데이터 분포가 국소(local)적으로도 매우 비선형적이면 local에서 선형성을 가정하는 LIME은 설명력에 한계를 갖게 됨
- 하이퍼파라미터에 따라서 샘플링 성능이 들쑥날쑥하는 불안정성(inconsistent)을 가짐
- 데이터 종류(이미지, 텍스트 등)나 surrogate 모델에 따라서 데이터 perturbation 방식을 다르게 해야하므로, model-agnostic 방법이 갖는 장점인 "유연성"을 다소 퇴색시킴

---

### SHAP (Shapley Additive exPlanations)
- Additive Feature Attribute methods (협업게임 이론 관점)
  - 각 팀원의 점수를 합하면 전체 점수가 된다. (공평) ➡️ `Additivity`
  - 매번 똑같은 방식으로 플레이했으나 개인의 게임점수가 측정되는 것이 다르다. (불공평) ➡️ `Consistency`
  - 팀플레이에 참여하지 않았는데 개인의 게임점수가 0이 아니다. (불공평) ➡️ `Missingness`
  - LIME에서 발생한 문제들(consistency, missingness) 해결
- Shapley Value를 직접적으로 계산하는 것은 모든 순열조합에 대해 체크해야하므로 계산량이 많다. ➡️ 효과적인 계산방법이 필요!

<br>

`-` SHAP variations

- KernelSHAP
  - Truly Model-Agnostic
  - Relatively Slow
  - Approximate calculation
- TreeSHAP
  - For Tree Models
  - Fast
  - Exact calculation
- DeepSHAP, GradientSHAP
  - For Deep Learning Models

<br>

`-` SHAP plot interpretation

- DeepSHAP : MNIST classification
- GradientSHAP : ImageNet classification
- TreeSHAP : NHANES Survival Model
  - summary plot (beeswarm plot)
  - dependency plot
  - interaction value plot
  - monitor plot

<br>

`-` 장점

- Model-Agnostic 방법론 중에서 Explanation model이 가져야할 좋은 특성들이 이론적으로 잘 증명됨
- 각 관측치에 대한 Local Explanation 뿐만 아니라, 각 feature 별 SHAP mean으로 Global Explanation도 얻을 수 있음

`-` 단점

- KernelSHAP의 경우, 속도가 느림
- 자칫하면 SHAP value를 원인 / 결과로 해석할 여지가 있음