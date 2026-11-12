import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# 配置
# ============================================================

REAL_DATA_PATH = "./LPD.csv"

# 固定划分结果
OUTPUT_SPLIT_PATH = "./real_data_split.csv"

# 保存带 split 标签的完整原始数据
OUTPUT_FULL_DATA_PATH = "./LPD_with_split.csv"

RANDOM_STATE = 42


# ============================================================
# 1. 读取原始真实数据
# ============================================================

print("=" * 70)
print("Step 1: Loading real-world dataset")
print("=" * 70)

df = pd.read_csv(REAL_DATA_PATH)

print(f"原始数据行数: {len(df)}")
print(f"原始列名: {df.columns.tolist()}")


# ============================================================
# 2. 检查必要字段
# ============================================================

required_columns = [
    "class_label",
    "sample_id"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"缺少必要列: {col}"
        )


# ============================================================
# 3. 创建 unique_id
# ============================================================

if "unique_id" not in df.columns:

    df["unique_id"] = (
        df["class_label"].astype(str)
        + "_"
        + df["sample_id"].astype(str)
    )


# ============================================================
# 4. 提取“样本级”信息

# ============================================================

sample_df = (
    df[
        [
            "unique_id",
            "class_label",
            "sample_id"
        ]
    ]
    .drop_duplicates(
        subset="unique_id"
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 70)
print("Step 2: Sample-level dataset")
print("=" * 70)

print(
    f"独立时间序列数量: "
    f"{len(sample_df)}"
)

print("\n各类别样本数:")

print(
    sample_df[
        "class_label"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 5. 第一次划分
#
# 60% → Train
# 40% → 临时集合
#
# 使用 stratify 保证类别比例基本一致
# ============================================================

train_df, temp_df = train_test_split(

    sample_df,

    test_size=0.40,

    random_state=RANDOM_STATE,

    stratify=sample_df[
        "class_label"
    ]
)


# ============================================================
# 6. 第二次划分
#
# 临时集合占总数据的 40%
#
# 再一分为二:
#
# 20% → Validation
# 20% → Test
# ============================================================

val_df, test_df = train_test_split(

    temp_df,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=temp_df[
        "class_label"
    ]
)


# ============================================================
# 7. 添加 split 标签
# ============================================================

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"

val_df["split"] = "validation"

test_df["split"] = "test"


# ============================================================
# 8. 合并最终划分表
# ============================================================

split_df = pd.concat(

    [
        train_df,
        val_df,
        test_df
    ],

    axis=0,

    ignore_index=True

)


# 为了方便检查，按照 split + class_label 排序

split_df = split_df.sort_values(

    by=[
        "split",
        "class_label",
        "sample_id"
    ]

).reset_index(drop=True)


# ============================================================
# 9. 检查是否存在样本重复
# ============================================================

print("\n" + "=" * 70)
print("Step 3: Checking split integrity")
print("=" * 70)


train_ids = set(
    train_df["unique_id"]
)

val_ids = set(
    val_df["unique_id"]
)

test_ids = set(
    test_df["unique_id"]
)


# 检查集合交叉

train_val_overlap = (
    train_ids & val_ids
)

train_test_overlap = (
    train_ids & test_ids
)

val_test_overlap = (
    val_ids & test_ids
)


print(
    f"Train ∩ Validation: "
    f"{len(train_val_overlap)}"
)

print(
    f"Train ∩ Test: "
    f"{len(train_test_overlap)}"
)

print(
    f"Validation ∩ Test: "
    f"{len(val_test_overlap)}"
)


if (
    len(train_val_overlap) > 0
    or
    len(train_test_overlap) > 0
    or
    len(val_test_overlap) > 0
):

    raise RuntimeError(
        "发现数据泄漏："
        "不同数据集之间存在重复样本！"
    )


# ============================================================
# 10. 检查样本总数
# ============================================================

total_split_samples = (

    len(train_df)
    +
    len(val_df)
    +
    len(test_df)

)


if total_split_samples != len(sample_df):

    raise RuntimeError(
        "划分后的样本总数"
        "与原始样本数不一致！"
    )


print(
    "\n✓ 不存在样本重叠"
)

print(
    "✓ 所有样本均已分配"
)


# ============================================================
# 11. 输出总体样本数
# ============================================================

print("\n" + "=" * 70)
print("Step 4: Final split sizes")
print("=" * 70)


print(
    f"Train:      "
    f"{len(train_df)} "
    f"({len(train_df) / len(sample_df) * 100:.2f}%)"
)

print(
    f"Validation: "
    f"{len(val_df)} "
    f"({len(val_df) / len(sample_df) * 100:.2f}%)"
)

print(
    f"Test:       "
    f"{len(test_df)} "
    f"({len(test_df) / len(sample_df) * 100:.2f}%)"
)


# ============================================================
# 12. 查看每个 split 的类别分布
# ============================================================

print("\n" + "=" * 70)
print("Step 5: Class distribution")
print("=" * 70)


distribution = pd.crosstab(

    split_df["class_label"],

    split_df["split"]

)


# 固定列顺序

desired_columns = [
    "train",
    "validation",
    "test"
]

distribution = distribution[
    [
        col
        for col in desired_columns
        if col in distribution.columns
    ]
]


distribution["Total"] = (
    distribution.sum(axis=1)
)


print(
    "\n各类别划分结果:"
)

print(
    distribution
)


# ============================================================
# 13. 输出每个 split 内部的类别比例
# ============================================================

print("\n" + "=" * 70)
print("Class proportions within each split")
print("=" * 70)


for split_name in [
    "train",
    "validation",
    "test"
]:

    sub = split_df[
        split_df["split"]
        == split_name
    ]

    print(
        f"\n[{split_name.upper()}]"
    )

    counts = (
        sub["class_label"]
        .value_counts()
        .sort_index()
    )

    ratios = (
        sub["class_label"]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    result = pd.DataFrame(
        {
            "N": counts,
            "Percentage": ratios
        }
    )

    result[
        "Percentage"
    ] = result[
        "Percentage"
    ].round(2)

    print(result)


# ============================================================
# 14. 保存固定划分文件
# ============================================================

split_df.to_csv(

    OUTPUT_SPLIT_PATH,

    index=False,

    encoding="utf-8-sig"

)


print("\n" + "=" * 70)
print("Step 6: Saving split file")
print("=" * 70)

print(
    f"✓ 固定划分文件已保存:\n"
    f"{OUTPUT_SPLIT_PATH}"
)


# ============================================================
# 15. 可选：
#     把 split 信息重新合并回原始逐时间点数据
#
# 后续代码可以直接读取这个文件，
# 每个时间点都带有 train / validation / test 标签。
# ============================================================

full_df = df.merge(

    split_df[
        [
            "unique_id",
            "split"
        ]
    ],

    on="unique_id",

    how="left"

)


# 检查是否有未匹配样本

if full_df["split"].isna().any():

    n_missing = (
        full_df["split"]
        .isna()
        .sum()
    )

    raise RuntimeError(
        f"有 {n_missing} 行"
        f"没有匹配到 split 标签"
    )


full_df.to_csv(

    OUTPUT_FULL_DATA_PATH,

    index=False,

    encoding="utf-8-sig"

)


print(
    f"\n✓ 带 split 标签的完整数据已保存:\n"
    f"{OUTPUT_FULL_DATA_PATH}"
)


# ============================================================
# 16. 最终检查
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)


print(
    f"原始独立样本数: "
    f"{len(sample_df)}"
)

print(
    f"Train样本数: "
    f"{len(train_df)}"
)

print(
    f"Validation样本数: "
    f"{len(val_df)}"
)

print(
    f"Test样本数: "
    f"{len(test_df)}"
)

print(
    f"总计: "
    f"{total_split_samples}"
)


assert (
    total_split_samples
    ==
    len(sample_df)
)


assert (
    len(train_ids & val_ids)
    ==
    0
)

assert (
    len(train_ids & test_ids)
    ==
    0
)

assert (
    len(val_ids & test_ids)
    ==
    0
)


print("\n" + "=" * 70)

print(
    "✅ 60% Train + 20% Validation "
    "+ 20% Test 固定划分完成"
)

print(
    "✅ 不存在样本重叠"
)

print(
    "✅ Test set 可以作为最终独立测试集"
)

print("=" * 70)