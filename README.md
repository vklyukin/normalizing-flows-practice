# normalizing-flows-practice

A minimal, self-contained implementation of the **Glow** normalizing flow
([Kingma & Dhariwal, 2018](https://arxiv.org/abs/1807.03039)), packaged for use in
a teaching notebook. It is a single Python package, `glow`, with no dependencies
beyond PyTorch. The same model serves both vector (tabular) data and images via a
single `is_1d` flag.

The code is extracted, unchanged, from the
[`nf_distillation`](https://github.com/vklyukin/nf_distillation) research
repository and reduced to only what the practice notebook needs.

---

## English

### Install

```bash
git clone https://github.com/vklyukin/normalizing-flows-practice.git
```

Then make the package importable (in Colab or locally):

```python
import sys
sys.path.insert(0, "normalizing-flows-practice")
from glow import Glow
```

The only requirement is `torch` (already available in Google Colab).

### Usage

`Glow` is a single `nn.Module`. Its forward pass returns a triple
`(z, bpd, y_logits)`:

- `z` — the latent code,
- `bpd` — the training loss per object: bits-per-dimension for images, and the
  negative log-likelihood (in nats) for 1-D vector data,
- `y_logits` — class logits, unused here (pass `y_condition=False`).

**Vector data (1-D), e.g. 2-D points:**

```python
import torch
from glow import Glow

model = Glow(
    image_shape=[2],          # number of features
    hidden_channels=64,
    K=8, L=1,                 # K flow steps, L levels
    actnorm_scale=1.0,
    flow_permutation="invconv",
    flow_coupling="affine",
    LU_decomposed=True,
    y_classes=0, learn_top=False, y_condition=False,
    is_1d=True,
)

x = torch.randn(512, 2)
z, bpd, _ = model(x)          # density evaluation
loss = bpd.mean()            # negative log-likelihood

# sampling: map Gaussian noise back to data space
latent_dim = model.flow.output_shapes[-1][1]
z = torch.randn(1000, latent_dim)
samples = model(z=z, reverse=True)
```

**Image data (2-D), e.g. 32×32×3:**

```python
model = Glow(
    image_shape=[32, 32, 3],  # [H, W, C]
    hidden_channels=128,
    K=8, L=3,
    actnorm_scale=1.0,
    flow_permutation="invconv",
    flow_coupling="affine",
    LU_decomposed=True,
    y_classes=0, learn_top=False, y_condition=False,
    is_1d=False,
)

x = torch.rand(64, 3, 32, 32)   # values in [0, 1]
z, bpd, _ = model(x)
loss = bpd.mean()              # bits per dimension

# sampling from the prior (temperature < 1 gives cleaner samples)
samples = model(reverse=True, temperature=0.7)
```

### Notes

- ActNorm layers initialize from the first mini-batch on the first forward pass —
  no manual initialization needed.
- During encoding, the multi-scale model keeps only the top-level latent; the
  split-off parts are re-sampled on decode. Exact image reconstruction is
  therefore not supported — interpolate between latent codes instead.

---

## Русский

### Установка

```bash
git clone https://github.com/vklyukin/normalizing-flows-practice.git
```

Затем подключите пакет (в Colab или локально):

```python
import sys
sys.path.insert(0, "normalizing-flows-practice")
from glow import Glow
```

Единственная зависимость — `torch` (уже установлен в Google Colab).

### Использование

`Glow` — это один `nn.Module`. Прямой проход возвращает тройку
`(z, bpd, y_logits)`:

- `z` — скрытое представление;
- `bpd` — функция потерь на объект: число бит на измерение для изображений и
  отрицательное логарифмическое правдоподобие (в натах) для одномерных
  (векторных) данных;
- `y_logits` — логиты классов, здесь не используются (`y_condition=False`).

**Векторные данные (одномерный режим), например точки на плоскости:**

```python
import torch
from glow import Glow

model = Glow(
    image_shape=[2],          # число признаков
    hidden_channels=64,
    K=8, L=1,                 # K шагов потока, L уровней
    actnorm_scale=1.0,
    flow_permutation="invconv",
    flow_coupling="affine",
    LU_decomposed=True,
    y_classes=0, learn_top=False, y_condition=False,
    is_1d=True,
)

x = torch.randn(512, 2)
z, bpd, _ = model(x)          # оценка плотности
loss = bpd.mean()            # отрицательное правдоподобие

# генерация: переводим гауссовский шум в пространство данных
latent_dim = model.flow.output_shapes[-1][1]
z = torch.randn(1000, latent_dim)
samples = model(z=z, reverse=True)
```

**Изображения (двумерный режим), например 32×32×3:**

```python
model = Glow(
    image_shape=[32, 32, 3],  # [H, W, C]
    hidden_channels=128,
    K=8, L=3,
    actnorm_scale=1.0,
    flow_permutation="invconv",
    flow_coupling="affine",
    LU_decomposed=True,
    y_classes=0, learn_top=False, y_condition=False,
    is_1d=False,
)

x = torch.rand(64, 3, 32, 32)   # значения в диапазоне [0, 1]
z, bpd, _ = model(x)
loss = bpd.mean()              # бит на измерение

# генерация из априорного распределения (temperature < 1 — чище сэмплы)
samples = model(reverse=True, temperature=0.7)
```

### Примечания

- Слои ActNorm инициализируются по первому мини-батчу при первом проходе вперёд —
  ручная инициализация не нужна.
- При кодировании многомасштабная модель сохраняет только латент верхнего уровня;
  отделённые на промежуточных уровнях части заново сэмплируются при декодировании.
  Поэтому точная реконструкция изображений не поддерживается — вместо этого
  интерполируйте между латентными кодами.

---

## Acknowledgements / Благодарности

Glow was introduced by Kingma & Dhariwal (2018). This implementation is taken from
the [`nf_distillation`](https://github.com/vklyukin/nf_distillation) repository.
