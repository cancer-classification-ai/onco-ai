"""제출까지 한 번에 가는 파이프라인.

fold → 피처 파켓 → seed 앙상블 학습 → 가중 블렌드 + 로짓 보정 → 짝 라벨 규칙 → 제출.
`scripts/run_pipeline.py` 와 `notebooks/13_pipeline.ipynb` 가 **둘 다 이 모듈을 부른다** —
진입점이 둘이어도 코드 경로는 하나여야 결과가 어긋나지 않는다.

재현 가능성이 설계 조건이다
---------------------------
이 대회는 여러 사람이 각자 뽑은 OOF 를 모아 블렌딩한다. 같은 설정이 같은 숫자를 내지
못하면 각자의 CV 는 멀쩡한데 합친 결과만 조용히 어긋난다. 그래서:

- 난수원을 전부 고정한다 (모델 seed · fold seed · 잠재 · 모듈 · 그리디).
- **`gpu_ram_part` 를 숫자로 박는다.** `auto` 는 실행 시점의 GPU 여유로 값을 정해서
  같은 seed 로도 CatBoost 결과가 갈린다 — 실측으로 6,201행 중 56행의 라벨이 달랐다.
  고정하면 비트 단위로 같아진다. 자세한 건 `docs/reproducibility.md`.
- 실행마다 `run.json` 에 설정·환경·파켓 지문·점수를 남긴다.

무엇을 고정으로 두었나
----------------------
피처셋 `f16`, 결합 가중 `0.45/0.45/0.10`, 짝 규칙 `min_mut=3` 은 LB 로 검증된 값이라
기본값으로 둔다. 바꾸려면 `PipelineConfig` 를 고치면 되지만, 바꾼 값은 LB 근거가 없다.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import MacroF1LogitBias
from .ensemble import crossfit_calibrated_blend, weighted_average
from .metrics import macro_f1, read_prediction_frame
from .pair_rule import PAIR, build_pair_rule
from .paths import artifacts_dir, process_dir, raw_dir, use_run_dirs
from .provenance import check_fold_fingerprint, fold_fingerprint, parquet_fingerprint
from .runlog import RunLog
from .validation import SEED as FOLD_SEED, SEED_ENSEMBLE

__all__ = ["PipelineConfig", "run_pipeline", "FEATURE_COMMANDS", "SEED_ENSEMBLE"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: `f16` 16블록이 읽는 소스를 만드는 `make_features.py` 호출. (서브커맨드, 추가 인자).
#: train/test 를 따로 돌린다 — 행마다 독립 계산이라 train 통계가 test 로 새지 않는다.
FEATURE_COMMANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("domain", ()),                             # domain
    ("sample", ("--include-cell-rollup",)),     # rollup · rollup16
    ("enc3", ()),                               # enc3 · comut · lsvd · lnmf · gmod
    ("gene", ("--kind", "event_count")),        # gec
    ("gene", ("--kind", "mutated")),            # ebovr 계열
    ("gene-types", ()),                         # gtype
    ("parsed", ()),                             # parsed19
    ("burden-extra", ()),                       # burden8
    ("amino", ()),                              # aa9
    ("sigtokens", ()),                          # sigtok
    ("tokens", ()),                             # exacttok
    ("parsed-tokens", ()),                      # ptok
)


@dataclass(frozen=True)
class PipelineConfig:
    """한 번의 실행을 정하는 값 전부. `run.json` 에 그대로 기록된다."""

    run_tag: str

    #: seed 앙상블. 모델마다 이 seed 로 각각 학습하고 전부 블렌드에 넣는다.
    #: 값은 `cancer_hack.validation.SEED_ENSEMBLE` 한 곳에서 온다 — 흩어 두면
    #: 사람마다 다른 seed 로 돌리게 되고 그 순간 OOF 를 못 섞는다.
    seeds: tuple[int, ...] = SEED_ENSEMBLE
    #: fold 분할 seed. 모델 seed 와 별개다.
    fold_seed: int = FOLD_SEED

    feature_config: str = "f16"
    cv: str = "sgkf"                 # fold_group5 — 같은 프로파일을 한 fold 로 묶는다
    n_splits: int = 5
    topk: int = 500

    #: 모델별 가중. seed 수로 나눠서 소스마다 배분한다.
    model_weights: Mapping[str, float] = field(
        default_factory=lambda: {"xgb": 0.45, "catboost": 0.45, "rf": 0.10}
    )
    #: 모델별 `train_gbdt.PARAM_PRESETS` 이름. 비우면 기본 하이퍼파라미터.
    params_presets: Mapping[str, str] = field(default_factory=dict)

    min_mut: int = 3                 # 짝 라벨 규칙
    apply_pair_rule: bool = True     # 끄지 않는다 — LB +0.09 다

    #: **숫자로 고정한다.** `auto` 는 실행 시점의 GPU 여유로 값을 정해 설정 자체가
    #: 매번 달라진다. 재현이 목적이면 숫자여야 한다.
    gpu_ram_part: float = 0.4

    #: 기본 device. XGBoost·RandomForest 는 GPU 에서도 비트 단위로 재현된다.
    device: str = "auto"

    #: **모델별 device 예외.** CatBoost GPU 학습은 같은 seed·같은 설정으로도 결과가
    #: 간헐적으로 갈린다 — f2l 로 3회 돌렸을 때 한 번이 OOF 최대차 7.9e-02 로 어긋났다.
    #: `device` 와 `gpu_ram_part` 를 둘 다 숫자로 박아도 그렇다. CPU 로 돌리면 OOF 도
    #: test 도 비트 단위로 같다(실측). 제출본을 만드는 자리에서는 재현이 속도보다 앞선다.
    #:
    #: 빠른 탐색이 필요하면 `device_by_model={}` 로 비우고 GPU 를 쓰되, 그렇게 뽑은
    #: OOF 는 "다시 만들면 달라질 수 있는 것"으로 다룬다. `docs/reproducibility.md` 참고.
    device_by_model: Mapping[str, str] = field(
        default_factory=lambda: {"catboost": "cpu"}
    )

    rebuild_folds: bool = True
    rebuild_features: bool = True
    train_models: bool = True

    #: 대조할 기준 제출 csv. 있으면 전 행 비교한다.
    reference: Path | None = None

    def as_dict(self) -> dict:
        return {
            "run_tag": self.run_tag,
            "seeds": list(self.seeds),
            "fold_seed": self.fold_seed,
            "feature_config": self.feature_config,
            "cv": self.cv,
            "n_splits": self.n_splits,
            "topk": self.topk,
            "model_weights": dict(self.model_weights),
            "params_presets": dict(self.params_presets),
            "min_mut": self.min_mut,
            "apply_pair_rule": self.apply_pair_rule,
            "gpu_ram_part": self.gpu_ram_part,
            "device": self.device,
            "device_by_model": dict(self.device_by_model),
        }

    @property
    def models(self) -> list[str]:
        return list(self.model_weights)

    @property
    def source_weights(self) -> list[float]:
        """(모델 × seed) 순서의 가중. 모델 가중을 seed 수로 나눈다."""
        return [self.model_weights[m] / len(self.seeds) for m in self.models
                for _ in self.seeds]


# ------------------------------------------------------------------ 단계
def ensure_folds(config: PipelineConfig, log: RunLog) -> Path:
    from .io import save_parquet
    from .validation import build_fold_frame

    path = process_dir() / "train_folds.parquet"
    if config.rebuild_folds or not path.exists():
        with log.step("fold 생성"):
            folds = build_fold_frame(
                raw_dir() / "train.csv",
                label_column="SUBCLASS",
                n_splits=config.n_splits,
                seed=config.fold_seed,
                group_cache_path=process_dir() / "train_group_keys.parquet",
            )
            save_parquet(folds, path)
            log.artifact("folds", path)
    log.record("fold_fingerprint", fold_fingerprint(path))
    return path


def ensure_features(config: PipelineConfig, log: RunLog) -> dict[str, str]:
    if config.rebuild_features:
        with log.step("피처 파켓 생성"):
            for command, extra in FEATURE_COMMANDS:
                for split in ("train", "test"):
                    _make_features(command, extra, split)
    with log.step("파켓 지문"):
        prints = {p.name: parquet_fingerprint(p)
                  for p in sorted(process_dir().glob("*.parquet"))}
        log.record("feature_fingerprints", prints)
        log.record("n_feature_parquets", len(prints))
    return prints


def _make_features(command: str, extra: Sequence[str], split: str) -> None:
    """`make_features.py` 를 서브프로세스로. 환경변수가 상속돼 출력 위치가 따라간다."""
    argv = [sys.executable, str(PROJECT_ROOT / "scripts/make_features.py"),
            command, "--split", split, "--overwrite", *extra]
    done = subprocess.run(argv, capture_output=True, text=True,
                          encoding="utf-8", cwd=PROJECT_ROOT)
    if done.returncode != 0:
        raise RuntimeError(
            f"make_features {command} {split} 실패\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
        )


def train_all(config: PipelineConfig, log: RunLog) -> dict[tuple[str, int], str]:
    """모델 × seed 를 전부 학습하고 stem 을 돌려준다."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from train_gbdt import CONFIGS, Dataset, build_parser, run_config

    stems: dict[tuple[str, int], str] = {}
    if not config.train_models:
        for model in config.models:
            for seed in config.seeds:
                stems[(model, seed)] = _find_cached(config, model, seed)
        return stems

    with log.step("Dataset 준비"):
        data = Dataset(set(CONFIGS[config.feature_config]["blocks"]), n_splits=config.n_splits)

    for model in config.models:
        for seed in config.seeds:
            with log.step(f"학습 {model} seed {seed}"):
                args = build_parser().parse_args([])
                args.model = model
                args.topk = config.topk
                args.n_splits = config.n_splits
                args.seed = seed
                # CatBoost 만 CPU 로 내린다 — GPU 학습이 간헐적으로 다른 값을 낸다.
                args.device = config.device_by_model.get(model, config.device)
                args.override = {}
                args.params_preset = config.params_presets.get(model)
                args.dry_run = False
                args.submission = False
                args.tag = config.run_tag
                # 숫자로 고정한다. `auto` 면 같은 seed 로도 CatBoost 가 갈린다.
                args.gpu_ram_part = config.gpu_ram_part
                result = run_config(data, config=config.feature_config, cv=config.cv, args=args)
                stems[(model, seed)] = result["stem"]
                log.record(f"oof_{model}_s{seed}", result["oof_macro_f1"])
    return stems


def _find_cached(config: PipelineConfig, model: str, seed: int) -> str:
    slug = {"skf": "skf5", "sgkf": "group5"}[config.cv]
    pattern = (f"oof_{model}_{config.run_tag}_{config.feature_config}_{slug}"
               f"_k{config.topk}_*_s{seed}.csv")
    hits = sorted((artifacts_dir() / "oof").glob(pattern))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern} 에 맞는 캐시가 {len(hits)}개다")
    return hits[0].stem[len("oof_"):]


def blend(config: PipelineConfig, stems: Mapping[tuple[str, int], str], log: RunLog):
    """9소스(모델 3 × seed 3) 고정 가중 블렌드 + 교차적합 로짓 보정."""
    with log.step("앙상블"):
        keys = [(m, s) for m in config.models for s in config.seeds]
        oof_frames = [_load(artifacts_dir() / "oof", "oof", stems[k]) for k in keys]
        test_frames = [_load(artifacts_dir() / "test_predictions", "test", stems[k])
                       for k in keys]

        classes = oof_frames[0][1]
        if any(c != classes for _, c in oof_frames + test_frames):
            raise ValueError("소스마다 클래스 구성이 다르다")
        columns = [f"p_{c}" for c in classes]

        oof = [f[columns].to_numpy(dtype=np.float64) for f, _ in oof_frames]
        test = [f[columns].to_numpy(dtype=np.float64) for f, _ in test_frames]
        y = oof_frames[0][0]["y_true"].to_numpy()
        ids = oof_frames[0][0]["ID"].astype(str).to_numpy()
        test_ids = test_frames[0][0]["ID"].astype(str).to_numpy()

        folds = pd.read_parquet(process_dir() / "train_folds.parquet")
        fold_column = f"fold_{'group5' if config.cv == 'sgkf' else 'skf5'}"
        fold_ids = (folds.set_index(folds["ID"].astype(str))
                    .reindex(ids)[fold_column].to_numpy())

        weights = config.source_weights
        raw_blend, crossfit, _ = crossfit_calibrated_blend(oof, y, classes, fold_ids, weights)
        class_array = np.asarray(classes)

        log.record("oof_uniform", macro_f1(
            y, class_array[weighted_average(oof, [1 / len(oof)] * len(oof)).argmax(1)]))
        log.record("oof_blend_raw", macro_f1(y, class_array[raw_blend.argmax(1)]))
        score = macro_f1(y, class_array[crossfit.argmax(1)])
        log.record("oof_blend_calibrated", score)

        # test 에는 정답이 없어 교차적합을 못 한다. 전체 OOF 로 맞춘 바이어스를 쓴다.
        bias = MacroF1LogitBias().fit(raw_blend, y, classes)
        test_proba = bias.predict_proba(weighted_average(test, weights))

    return {"classes": classes, "test_ids": test_ids, "test_proba": test_proba,
            "oof_macro_f1": score}


def _load(directory: Path, prefix: str, stem: str):
    frame, classes = read_prediction_frame(directory / f"{prefix}_{stem}.csv")
    return frame.sort_values("ID", kind="stable"), classes


def write_submissions(config: PipelineConfig, blended: dict, log: RunLog) -> dict[str, Path]:
    sample = pd.read_csv(raw_dir() / "sample_submission.csv")
    classes = np.asarray(blended["classes"])
    base = pd.DataFrame({"ID": blended["test_ids"],
                         "SUBCLASS": classes[blended["test_proba"].argmax(1)]})
    base = sample[["ID"]].astype(str).merge(base, on="ID", validate="one_to_one")

    out: dict[str, Path] = {}
    out["base"] = _write(base, artifacts_dir() / "submissions" / "submission_base.csv",
                         sample, set(blended["classes"]), log, "submission_base")

    if config.apply_pair_rule:
        with log.step("짝 규칙"):
            rule = build_pair_rule(raw_dir() / "train.csv", raw_dir() / "test.csv",
                                   min_mut=config.min_mut)
            violations = rule.verify_premises()
            if violations:
                raise ValueError("짝 규칙 전제가 깨졌다: " + " · ".join(violations))
            before = base["SUBCLASS"].to_numpy().copy()
            final = base.copy()
            final["SUBCLASS"] = rule.relabel(final["ID"], before)
            log.record("pair_rule_rows", rule.diagnostics["n_flipped"])
            log.record("pair_rule_changed",
                       int((final["SUBCLASS"].to_numpy() != before).sum()))
            log.record("pair_rule_was_copying_train_label", sum(
                1 for i, c in zip(final["ID"], before)
                if i in rule.mapping and PAIR.get(c) == rule.mapping[i]))
            log.record("pair_rule_diagnostics", rule.diagnostics)
        out["final"] = _write(
            final, artifacts_dir() / "submissions" / f"submission_pairrule_m{config.min_mut}.csv",
            sample, set(blended["classes"]), log, "submission_pairrule")
    return out


def _write(frame: pd.DataFrame, path: Path, sample: pd.DataFrame,
           classes: set[str], log: RunLog, label: str) -> Path:
    if list(frame.columns) != ["ID", "SUBCLASS"]:
        raise ValueError(f"제출 컬럼이 ['ID','SUBCLASS'] 가 아니다: {list(frame.columns)}")
    if len(frame) != len(sample):
        raise ValueError(f"행 수 {len(frame)}, 기대 {len(sample)}")
    if not (frame["ID"].astype(str).to_numpy() == sample["ID"].astype(str).to_numpy()).all():
        raise ValueError("ID 순서가 sample_submission 과 다르다")
    if frame["SUBCLASS"].isna().any() or not set(frame["SUBCLASS"]) <= classes:
        raise ValueError("라벨이 비었거나 클래스 밖이다")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="UTF-8-sig")
    log.artifact(label, path)
    return path


def compare_reference(reference: Path, produced: Path, log: RunLog) -> int | None:
    if not reference.exists():
        return None
    a, b = pd.read_csv(reference), pd.read_csv(produced)
    if not (a["ID"].astype(str).to_numpy() == b["ID"].astype(str).to_numpy()).all():
        raise ValueError("기준 파일과 ID 순서가 다르다")
    n_diff = int((a["SUBCLASS"].to_numpy() != b["SUBCLASS"].to_numpy()).sum())
    log.record("reference", str(reference))
    log.record("reference_rows_differ", n_diff)
    return n_diff


# ------------------------------------------------------------------ 진입점
def run_pipeline(config: PipelineConfig) -> dict:
    """전체를 돌리고 `run.json`·`run.md` 를 남긴다."""
    dirs = use_run_dirs(config.run_tag)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        if not (dirs["raw"] / name).exists():
            raise FileNotFoundError(f"{name} 이 {dirs['raw']} 에 없다")

    log = RunLog(dirs["artifacts"], run_tag=config.run_tag, config=config.as_dict())
    ensure_folds(config, log)
    # fold 가 기준선과 같은지 — 다르면 기존 OOF 와 섞을 수 없다. 새로 만들었으면 경고만.
    try:
        check_fold_fingerprint(process_dir() / "train_folds.parquet")
        log.record("fold_matches_baseline", True)
    except ValueError:
        log.record("fold_matches_baseline", False)

    ensure_features(config, log)
    stems = train_all(config, log)
    log.record("stems", {f"{m}_s{s}": v for (m, s), v in stems.items()})
    blended = blend(config, stems, log)
    paths = write_submissions(config, blended, log)

    if config.reference is not None and "final" in paths:
        compare_reference(config.reference, paths["final"], log)

    written = log.save()
    return {"config": config, "log": log, "paths": paths,
            "oof_macro_f1": blended["oof_macro_f1"], "run_files": written,
            "dirs": dirs}
