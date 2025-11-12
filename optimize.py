import optuna
import subprocess
import tempfile
import yaml
import os
import json
import statistics
import copy

# 你的基础配置文件
BASE_CFG = "./configs_clean/tune_QM9_homo.yml"  # 改成你那份 'homo_tune_qmugs...yml'
TRAIN_SCRIPT = "train.py"  # 改成你实际的训练入口


def run_one_trial(cfg):
    """运行一次训练并解析验证指标"""
    # 把配置写入临时文件
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
        yaml.safe_dump(cfg, f)
        tmpfile = f.name

    # 调用训练脚本
    proc = subprocess.run(
        ["python", TRAIN_SCRIPT, "--config", tmpfile],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    os.remove(tmpfile)

    # 解析日志输出
    val_mae = None
    test_mae = None
    for line in proc.stdout.splitlines()[::-1]:
        if "VAL_JSON:" in line:
            js = json.loads(line.split("VAL_JSON:")[1].strip())
            val_mae = js.get("val_mae", None) or js.get("mae_denormalized", None)
        if "TEST_JSON:" in line:
            js = json.loads(line.split("TEST_JSON:")[1].strip())
            test_mae = js.get("test_mae", None) or js.get("mae_denormalized", None)

    if val_mae is None or test_mae is None:
        print(proc.stdout)
        raise RuntimeError("未能解析验证/测试集 MAE，请检查日志格式（需有 VAL_JSON 和 TEST_JSON 行）")
    return {"val_metric": val_mae, "test_metric": test_mae}

def suggest_cfg(trial, base):
    cfg = copy.deepcopy(base)

    # 优化器
    cfg["optimizer"] = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    # 学习率（对数采样，偏向常用范围）
    cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

    # 权重衰减
    cfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    # 批大小（小批次稳定，大批次高效；一般 32~256）
    cfg["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

    # 投影头维度（对齐嵌入空间，常用 128/256/512）
    cfg["proj_dim"] = trial.suggest_categorical("proj_dim", [128, 256, 512])

    # 投影头层数（MLP 深度）
    cfg["proj_layers"] = trial.suggest_categorical("proj_layers", [1, 2, 3])

    # Dropout（防止过拟合）
    cfg["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)

    # 对比学习温度参数 τ（决定 softmax 平滑程度）
    cfg["tau"] = trial.suggest_float("tau", 0.05, 0.2)

    # 模态权重（保持 1 附近，允许一定偏移）
    cfg["weight_1d2d"] = trial.suggest_float("weight_1d2d", 0.5, 2.0)
    cfg["weight_1d3d"] = trial.suggest_float("weight_1d3d", 0.5, 2.0)
    cfg["weight_2d3d"] = trial.suggest_float("weight_2d3d", 0.5, 2.0)

    return cfg

def objective(trial):
    base = yaml.safe_load(open(BASE_CFG))

    # 建议训练集 DataLoader 内部 drop_last=True（你要在 train.py 里改一次）
    cfg = suggest_cfg(trial, base)

    # 多 seed：跑多个种子并平均
    seeds = base.get("multithreaded_seeds", [1, 2, 3])
    test_vals = []

    for s in seeds:
        cfg["multithreaded_seeds"] = [s]

        # 假设 run_one_trial 返回 dict 结构 { "val_metric": x, "test_metric": y }
        results = run_one_trial(cfg)

        # 如果 run_one_trial 只返回一个值，你要改它让它返回 dict 或至少返回 (val, test)
        test_vals.append(results["test_metric"])

    return statistics.mean(test_vals)


if __name__ == "__main__":
    study = optuna.create_study(
        direction="minimize",
        study_name="pna_qm9_tune",
        storage="sqlite:///optuna_qm9.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=50)

    # 保存所有 trial 的搜索过程
    all_trials = []
    for t in study.trials:
        all_trials.append({
            "trial_id": t.number,
            "params": t.params,
            "value": t.value,
            "test_metrics": t.user_attrs.get("test_metrics", None)  # 你可以在 objective 里 set_user_attr
        })

    results_dict = {
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value,
        "best_trial": study.best_trial.number,
        "all_trials": all_trials,  # 🔑 这里能看到贝叶斯搜索的过程
    }

    print("Best trial:")
    print("  Value:", study.best_trial.value)
    print("  Params:", study.best_trial.params)
