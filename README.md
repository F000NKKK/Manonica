# Manonica

Manonica — экспериментальный физический движок магической инженерии для
Minecraft: свободные и связанные маноны, распределение потенциала, перенос,
энергетика и математически определённые конструкции. Магические эффекты
возникают из уравнений и свойств материалов, не из списка жёстко заданных
заклинаний.

**Состояние:** исследовательский прототип, не готовый PhysX-аналог и не
готовая Minecraft-симуляция. Первая целевая Minecraft-интеграция — Fabric
26.3; адаптер пока не управляет полноценной физикой мира.

## Слои

| Crate | Область ответственности |
| --- | --- |
| `manonica-error` | Типизированные ошибки |
| `manonica-core` | Единицы, количества, линейная алгебра, время |
| `manonica-physics` | Свободные маноны, 3D-сетка, диффузия, адвекция, потенциал, дискретные граничные условия, экспериментальная спектральная механика и интерфейс backend |
| `manonica-materials` | Связанные маноны, `TrapMaterial`, `MaterialLattice`, теплопроводность, каналы и тензорная анизотропия |
| `manonica-simulation` | Связанный шаг физики, состояние материалов и энергетический баланс; общий контракт CPU/GPU для Lab, симулятора и мода |
| `manonica-control` | Типизированное управление, P/I/D, диагностика и модели ТАУ |
| `manonica-compute-gpu` | Резидентный WGPU backend: полный связанный физический шаг, параллельные операторы, свёртки и атомарные снимки |
| `manonica-world` | Игровой runtime: JSONL-реестры, блоки, предметы, сущности, инвентари, чанки и NBT-сохранения |
| `manonica-simulator` | Консольные воспроизводимые опыты |
| `manonica-lab` | Конфигурирование опытов, редактор формул, рендер Meridian SDK и F3 |
| `adapters/minecraft/fabric/mc26_3` | Адаптер игры; законы не живут в Java/Fabric |

**Направление зависимостей:** `core` и `error` не зависят от физических
доменов; `physics` не зависит от `materials`, `Lab`, `wgpu` или
Minecraft. `materials -> physics`, `simulation -> materials + physics`,
`compute-gpu -> simulation`, а Lab компонует конфигурацию и отображение. Это исключает циклы зависимостей и позволяет
подключать вычислитель к будущему моду.

Архитектурные решения: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Правила проекта следуют Craftbook v2: границы по ответственности и
измеряемые инварианты, без искусственного разделения каждого типа на crate.

## Вычислительные backend

Есть только два backend: **CPU** и **GPU**. `auto` (по умолчанию)
сначала пробует GPU, затем CPU; `--backend gpu` требует GPU и завершает
приложение с ошибкой при его отказе; `--backend cpu` полностью исключает
GPU-вычисления.

GPU-режим Lab выполняет диффузию, однородный и вихревой перенос,
потенциальный дрейф, связывание, каналы, тепло и энергетический учёт на
устройстве. Решение потенциала также выполняется GPU-решателем.
Состояние остаётся резидентным между пакетами; снимки считываются для
наблюдения, а не между отдельными операторами. CPU готовит геометрию и
проверяет снимки; независимый CPU-решатель служит эталоном и резервом.

Потоки и свёртки параллельны. Каналы и тепловые обмены группируются по
независимым рёбрам с сохранением порядка зависимых операций. В F3 видны
время полного GPU-пакета, число его шагов и время рендера с ожиданиями.
Подробности: [docs/compute-backends.md](docs/compute-backends.md).

## Быстрый запуск

Нужен актуальный stable Rust. Базовые библиотеки проверяются отдельно
от графических зависимостей:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings

cargo test --manifest-path crates/manonica-compute-gpu/Cargo.toml
cargo test --manifest-path apps/manonica-lab/Cargo.toml
```

Запуск лаборатории (GPU первым, CPU при недоступности):

```bash
cargo run --release --manifest-path apps/manonica-lab/Cargo.toml -- \
  apps/manonica-lab/experiments/formula_comparison.toml
```

Строгий выбор устройства:

```bash
cargo run --release --manifest-path apps/manonica-lab/Cargo.toml -- \
  apps/manonica-lab/experiments/local_boundary_conductor.toml --backend gpu
```

Эксперимент с геометрически управляемым полем:
`potential_concentrator.toml`; трёхконтурное сравнение:
`formula_comparison.toml`. Формулы траекторий `x(t), y(t), z(t)`
растеризуются **одинаково** для физического источника, ловушки и меша.
Изменение геометрии должно изменять поле, а не только изображение.

Численные проверки без окна выполняются в `manonica-simulator`. Для
сравнения пакетной GPU-диффузии и однородного переноса с CPU-эталоном:

```bash
cargo run -p manonica-simulator --features gpu --bin gpu-transport
```

Полный связанный эксперимент, включая материалы, тепло и энергобаланс:

```bash
cargo run --release -p manonica-simulator --features gpu --bin gpu-coupled -- \
  --grid 32 24 32 --steps 120 --batch 3
```

Нужен Vulkan/WGPU compute adapter. Команда проверяет ошибки по ячейкам и
сохранение манонов; программный Vulkan пригоден для корректности, а замеры
ускорения требуют целевого оборудования. Наглядные 3D-опыты находятся в Lab.
Стиль и границы зависимостей задаёт `CraftBook.toml`; `cargo craftcheck check`
проверяет политику, `cargo craftcheck deps` — зависимости. Организация модулей
также проверяется ревью: checker не анализирует архитектуру исходного Rust.

## Качество и ограничения

Каждый новый закон требует симуляционного теста его **уравнения** и
отдельных проверок сохранения манонов и энергии.
GPU-результат сверяется с CPU-эталоном на доступном устройстве; обычная
CI без GPU не подтверждает аппаратную численную эквивалентность.

Для Minecraft-адаптера требуются Java 25 и Gradle 9.6:

```bash
cd adapters/minecraft/fabric/mc26_3
gradle build
```

См. [дорожную карту](docs/ROADMAP.md), [контроль законов](docs/physical-law-validation.md)
и [документацию Lab](apps/manonica-lab/README.md).

Экспериментальная динамическая механика доступна отдельным CPU API
`manonica_physics::SpectralMechanics` и GPU backend `GpuSpectral`.
В Lab доступен отдельный опыт `dynamic_mechanics.toml`. Стенд и контракт: [механический импульс](docs/mechanical-momentum.md).

```bash
cargo run --release -p manonica-simulator --bin spectral_field
```

Headless-демка игрового мира: [запуск, формат контента и ограничения](docs/game-runtime.md).

[Игровая 3D-демка: запуск, управление и текущие ограничения](docs/playable-demo.md).
