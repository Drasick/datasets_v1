import sys
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json

# 确保可以导入 datasets_v1.datasets_summary
WORKSPACE_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = WORKSPACE_ROOT / "datasets_v1"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.append(str(WORKSPACE_ROOT))

DATASETS_JSON_PATH = WORKSPACE_ROOT / "datasets_v1/datasets_summary.json"

def load_dataset(dataset_path):
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            # 更新
            st.session_state["datasets_summary"] = json.load(f)
    except Exception as import_error:  # pragma: no cover
        st.error(f"无法读取数据集摘要：{import_error}")
        st.stop()


def to_dataset_rows(summary: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ds in summary:
        dataset_name = ds.get("dataset_name")
        update_time = ds.get("update_time")
        try:
            update_dt = datetime.strptime(update_time, "%Y-%m-%d") if update_time else None
        except Exception:
            update_dt = None
        rows.append(
            {
                "dataset_name": dataset_name,
                "dataset_version": ds.get("dataset_version"),
                "description": ds.get("description"),
                "paper_url": ds.get("paper_url"),
                "dataset_url": ds.get("dataset_url"),
                "github_url": ds.get("github_url"),
                "languages": ds.get("languages", []),
                "task_type": ds.get("task_type"),
                "tool_preference": ds.get("tool_preference", []),
                "num_total_samples": ds.get("num_total_samples", 0),
                "dataset_splits": ds.get("dataset_splits", []),
                "update_time": update_time,
                "update_dt": update_dt,
                "organization": ds.get("organization"),
            }
        )
    return pd.DataFrame(rows)


def to_split_rows(summary: List[Dict[str, Any]]) -> pd.DataFrame:
    # 检查参数是否遗漏，参考 datasets_summary.py 的 split 字段
    rows: List[Dict[str, Any]] = []
    for ds in summary:
        dataset_name = ds.get("dataset_name")
        base_info = {
            "dataset_name": dataset_name,
            "dataset_version": ds.get("dataset_version"),
            "languages": ds.get("languages", []),
            "task_type": ds.get("task_type"),
            "tool_preference": ds.get("tool_preference", []),
            "organization": ds.get("organization"),
            "update_time": ds.get("update_time"),
        }
        for sp in ds.get("dataset_splits", []) or []:
            status = (sp.get("status") or {}).get("run_progess") or {}
            processed = status.get("processed_samples", 0) or 0
            correct = status.get("correct_samples", 0) or 0
            num_samples = sp.get("num_samples", 0) or 0

            # 检查 split 字段，补充遗漏字段
            rows.append(
                {
                    **base_info,
                    "split_name": sp.get("split_name"),
                    "display_name": sp.get("display_name"),
                    "num_samples": num_samples,
                    "file_path": sp.get("file_path"),
                    "has_reference_answer": sp.get("has_reference_answer", False),
                    "has_reasoning_process": sp.get("has_reasoning_process", False),
                    "is_multimodal": sp.get("is_multimodal", False),
                    "originality": sp.get("originality", []),
                    "column_mapping": sp.get("column_mapping", {}),
                    "split_description": sp.get("description", "无描述"),
                    "processed_samples": processed,
                    "correct_samples": correct,
                    "correct_samples_path": status.get("correct_samples_path"),
                    "status_downloaded": (sp.get("status") or {}).get("downloaded", False),
                }
            )
    return pd.DataFrame(rows)


def get_unique_values(df: pd.DataFrame, column: str) -> List[Any]:
    values: List[Any] = []
    if column not in df.columns:
        return values
    series = df[column].dropna()
    if series.empty:
        return values
    if series.apply(lambda x: isinstance(x, list)).any():
        all_values: List[Any] = []
        for item in series:
            if isinstance(item, list):
                all_values.extend(item)
        values = sorted(pd.unique(pd.Series(all_values)).tolist())
    else:
        values = sorted(series.astype(str).unique().tolist())
    return values


def apply_filters(
    df: pd.DataFrame,
    languages: List[str],
    task_types: List[str],
    tools: List[str],
    organizations: List[str],
    originality: List[str],
    is_multimodal: Optional[bool],
    has_reference_answer: Optional[bool],
    has_reasoning_process: Optional[bool],
) -> pd.DataFrame:
    filtered = df.copy()

    if languages:
        filtered = filtered[filtered["languages"].apply(lambda lst: any(lang in (lst or []) for lang in languages))]

    if task_types:
        filtered = filtered[filtered["task_type"].astype(str).isin(task_types)]

    if tools:
        filtered = filtered[filtered["tool_preference"].apply(lambda lst: any(t in (lst or []) for t in tools))]

    if organizations:
        filtered = filtered[filtered["organization"].astype(str).isin(organizations)]

    if originality and "originality" in filtered.columns:
        filtered = filtered[filtered["originality"].apply(lambda lst: any(o in (lst or []) for o in originality))]

    def maybe_filter_bool(column: str, value: Optional[bool]) -> None:
        nonlocal filtered
        if value is not None and column in filtered.columns:
            filtered = filtered[filtered[column] == value]

    maybe_filter_bool("is_multimodal", is_multimodal)
    maybe_filter_bool("has_reference_answer", has_reference_answer)
    maybe_filter_bool("has_reasoning_process", has_reasoning_process)

    return filtered


def render_metrics(datasets_df: pd.DataFrame, splits_df: pd.DataFrame) -> None:
    # 指标卡片样式（缩小尺寸）
    card_style = """
    <style>
    .kpi-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 8px 0 4px 0;
        margin-bottom: 0px;
        box-shadow: 0 1px 4px 0 rgba(0,0,0,0.04);
        text-align: center;
        height: 60px;
        min-height: 60px;
        max-height: 60px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-title {
        font-size: 13px;
        color: #666;
        margin-bottom: 2px;
        font-weight: 500;
    }
    .kpi-value {
        font-size: 22px;
        font-weight: bold;
        color: #222;
        margin-bottom: 0px;
    }
    .kpi-trend {
        font-size: 12px;
        font-weight: 500;
        margin-top: 2px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .kpi-trend-up {
        color: #27ae60;
    }
    .kpi-trend-down {
        color: #e74c3c;
    }
    .kpi-trend-flat {
        color: #888;
    }
    </style>
    """

    st.markdown(card_style, unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">📚 数据集数</div>
                <div class="kpi-value">{int(datasets_df.shape[0])}</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">🗂️ 切分(Split)数</div>
                <div class="kpi-value">{int(splits_df.shape[0])}</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col3:
        total_samples = int(datasets_df["num_total_samples"].fillna(0).sum()) if not datasets_df.empty else 0
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">🔢 样本总量</div>
                <div class="kpi-value">{total_samples:,}</div>
            </div>
            """, unsafe_allow_html=True
        )
    with col4:
        # 语言分布饼图
        lang_set = set()
        lg_map = {
            "zh": "中",
            "en": "英",
            "ja": "日",
            "lat": "拉丁",
        }
        for _, row in datasets_df.iterrows():
            langs = row.get("languages") or []
            for lg in langs:
                lang_set.add(lg)
        lang_list = sorted(list(lang_set))
        if lang_list:
            text = []
            for lg in lang_list:
                text.append(f"✅ {lg_map[lg]}  ")
            text_str = ' '.join(text)
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">💬 语言支持</div>
                <div class="kpi-value">{len(text)}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("当前筛选无语言信息")


def render_dataset_charts(datasets_df: pd.DataFrame, splits_df: pd.DataFrame, animate: bool = True) -> None:
    if datasets_df.empty:
        st.info("无数据用于可视化")
        return
    st.caption("")
    st.subheader("📊 样本量", help="🎋 样本数量基于 QA对 数据量进行统计")
    st.caption("注意：y 轴改为**对数刻度**，让几百和一万都能在图里清晰可见")
    # 数据集名和切分名缩写处理
    def short_name(name, max_len=15):
        name = str(name)
        return name if len(name) <= max_len else name[:max_len] + "..."

    def short_split_name(name, max_len=15):
        name = str(name)
        return name if len(name) <= max_len else name[:max_len] + "..."

    datasets_df = datasets_df.copy()
    datasets_df["dataset_name_short"] = datasets_df["dataset_name"].apply(short_name)

    # 按数据集绘制样本量柱状图，y轴使用对数坐标
    fig1 = px.bar(
        datasets_df.sort_values("num_total_samples", ascending=False),
        x="dataset_name_short",
        y="num_total_samples",
        color="organization",
        labels={"dataset_name_short": "数据集", "num_total_samples": "样本量", "organization": "组织"},
        hover_data=["dataset_name", "organization", "num_total_samples"],
        height=320,
    )
    fig1.update_layout(
        xaxis_title="数据集",
        yaxis_title="样本量",
        legend_title="组织",
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=14),
        yaxis_type="log",  # 设置y轴为对数坐标
        transition={"duration": 500, "easing": "cubic-in-out"} if animate else None,
    )
    st.plotly_chart(fig1, width='stretch', config={"displayModeBar": False})

    st.subheader("📦 切分(Split)样本量", help="🎋 样本数量基于 QA对 数据量进行统计")
    st.caption("注意：y 轴改为**对数刻度**，让几百和一万都能在图里清晰可见")
    if not splits_df.empty:
        # 堆叠柱状图
        agg = splits_df.groupby(["dataset_name", "display_name"], as_index=False)["num_samples"].sum()
        agg = agg.copy()
        agg["dataset_name_short"] = agg["dataset_name"].apply(short_name)
        agg["display_name_short"] = agg["display_name"].apply(short_split_name)
        fig2 = px.bar(
            agg,
            x="dataset_name_short",
            y="num_samples",
            color="display_name_short",
            labels={"dataset_name_short": "数据集", "num_total_samples": "样本量", "display_name_short": "切分"},
            height=320,
            hover_data=["dataset_name", "display_name", "num_samples"],  # 增加 hover data 显示完整信息
        )
        # 取消 legend 的自定义，使用默认
        fig2.update_layout(
            barmode="stack",
            xaxis_title="数据集",
            yaxis_title="样本量",
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="#fff",
            paper_bgcolor="#fff",
            font=dict(size=14),
            yaxis_type="log",  # 设置y轴为对数坐标
            transition={"duration": 500, "easing": "cubic-in-out"} if animate else None,
        )
        st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})

        # 新增：按 tool_preference 分类的饼状图
        st.subheader("🛠️ 工具偏好分布", help="⚠️ 粗略统计，同一条数据可能会在多个工具偏好中出现")
        import itertools
        import pandas as pd

        if "tool_preference" in datasets_df.columns:
            tool_rows = []
            for _, row in datasets_df.iterrows():
                tps = row.get("tool_preference", [])
                if not isinstance(tps, list):
                    continue
                for tp in tps:
                    tool_rows.append({
                        "dataset_name": row["dataset_name"],
                        "tool_preference": tp,
                        "num_total_samples": row.get("num_total_samples", 0)
                    })
            if tool_rows:
                tool_df = pd.DataFrame(tool_rows)
                # 按 tool_preference 分组统计样本总量
                agg_tool = tool_df.groupby("tool_preference", as_index=False)["num_total_samples"].sum()
                agg_tool["percent"] = agg_tool["num_total_samples"] / agg_tool["num_total_samples"].sum() * 100

                # 标记占比小于1%的工具
                agg_tool["tool_preference_display"] = agg_tool.apply(
                    lambda row: f"{row['tool_preference']} 🔍" if row["percent"] < 1 else row["tool_preference"], axis=1
                )

                fig4 = px.pie(
                    agg_tool,
                    names="tool_preference_display",
                    values="num_total_samples",
                    color="tool_preference_display",
                    labels={"tool_preference_display": "工具", "num_total_samples": "样本量"},
                    height=320,
                    hole=0.3,
                    hover_data=["tool_preference", "num_total_samples", "percent"]
                )
                fig4.update_traces(
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>样本量: %{value}<br>占比: %{percent:.2%}<extra></extra>"
                )

                # 饼图上只显示大于1%的标签，其余只在悬浮时显示
                fig4.update_traces(
                    texttemplate=[
                        f"{row['tool_preference']}: {row['percent']:.1f}%" if row["percent"] >= 1 else ""
                        for _, row in agg_tool.iterrows()
                    ]
                )

                fig4.update_layout(
                    margin=dict(l=20, r=20, t=30, b=20),
                    plot_bgcolor="#fff",
                    paper_bgcolor="#fff",
                    font=dict(size=14),
                    transition={"duration": 500, "easing": "cubic-in-out"} if animate else None,
                )
                st.plotly_chart(fig4, width='stretch', config={"displayModeBar": False})

                # 饼图下方增加说明
                if (agg_tool["percent"] < 1).any():
                    st.caption(
                        """
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1.5em;">🔍</span>
                            <span style="font-size: 1.1em;">部分工具占比低于 1%，仅在悬浮时显示。</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("当前筛选无工具偏好信息")
        else:
            st.info("当前筛选无工具偏好信息")


def render_split_charts(splits_df: pd.DataFrame, animate: bool = True) -> None:
    if splits_df.empty:
        st.info("无切分数据用于可视化")
        return
    st.caption("")
    st.subheader("📊 样本量", help="🎋 样本数量基于 QA对 数据量进行统计")
    st.caption("注意：y 轴改为**对数刻度**，让几百和一万都能在图里清晰可见")
    # display_name 只显示前15个字符，超出部分用...代替
    agg = splits_df.groupby(["display_name", "dataset_name"], as_index=False)["num_samples"].sum()
    agg["display_name_short"] = agg["display_name"].apply(lambda x: x[:15] + "..." if len(str(x)) > 10 else x)
    fig = px.bar(
        agg.sort_values("num_samples", ascending=False),
        x="display_name_short",
        y="num_samples",
        color="dataset_name",
        labels={"display_name_short": "切分", "num_samples": "样本量", "dataset_name": "数据集"},
        hover_data=["dataset_name", "display_name"],
        height=320,
    )
    fig.update_layout(
        xaxis_title="切分",
        yaxis_title="样本量",
        legend_title="数据集",
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="#fff",
        paper_bgcolor="#fff",
        font=dict(size=14),
        yaxis_type="log",  # 设置y轴为对数坐标
        transition={"duration": 500, "easing": "cubic-in-out"} if animate else None,
    )
    st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})


def render_progress(splits_df: pd.DataFrame) -> None:
    st.subheader("🚀 运行进度")
    if splits_df.empty:
        st.info("无切分进度可显示")
        return

    for _, row in splits_df.iterrows():
        ds = row["dataset_name"]
        sp = row["display_name"] or row.get("split_name")
        num_samples = int(row.get("num_samples", 0) or 0)
        processed = int(row.get("processed_samples", 0) or 0)
        correct = int(row.get("correct_samples", 0) or 0)
        log_path = row.get("correct_samples_path")
        progress = min(processed / num_samples, 1.0) if num_samples > 0 else 0
        acc = min(correct / processed, 1.0) if processed > 0 else 0

        st.markdown(f"**{ds} / {sp}**")
        col1, col2 = st.columns([2, 8])
        with col1:
            st.write(f"已处理 {processed:,} / {num_samples:,}（{progress*100:.2f}%）")
        with col2:
            st.progress(progress, text="处理进度")
        col3, col4 = st.columns([2, 8])
        with col3:
            st.write(f"正确 {correct:,} / 已处理 {processed:,}（{acc*100:.2f}%）")
        with col4:
            st.progress(acc, text="正确率")
        if isinstance(log_path, str) and log_path:
            st.caption(f"正确样本路径：{log_path}")
        st.divider()

@st.dialog("删除数据集", width="medium")
def delete_datasets_dialog():
    # 获取所有数据集名称
    ds_list = [ds.get("dataset_name", "") for ds in st.session_state.get("datasets_summary", [])]
    if not ds_list:
        st.info("当前没有可删除的数据集。")
        return

    # 使用checkbox选择要删除的数据集
    selected_to_delete = []
    with st.container(height=300):
        for ds_name in ds_list:
            if st.checkbox(f"删除：{ds_name}", key=f"delete_ds_checkbox_{ds_name}"):
                selected_to_delete.append(ds_name)
    confirm_text = st.text_input("请输入“我确定删除”以确认删除操作", key="delete_ds_confirm")

    if st.button("删除选中数据集", key="delete_ds_btn"):
        if not selected_to_delete:
            st.warning("请至少选择一个要删除的数据集。")
            return
        if confirm_text != "我确定删除":
            st.error("请正确输入“我确定删除”以确认删除。")
            return

        # 执行删除
        before_count = len(st.session_state["datasets_summary"])
        st.session_state["datasets_summary"] = [
            ds for ds in st.session_state["datasets_summary"]
            if ds.get("dataset_name", "") not in selected_to_delete
        ]
        after_count = len(st.session_state["datasets_summary"])
        try:
            with open(DATASETS_JSON_PATH, "w", encoding="utf-8") as f:
                import json
                json.dump(st.session_state["datasets_summary"], f, ensure_ascii=False, indent=4)
            st.toast(f"成功删除 {before_count - after_count} 个数据集，已写入！", icon="🗑️")
            time.sleep(1)
        except Exception as e:
            st.error(f"写入本地文件失败：{e}")

        time.sleep(1)
        st.rerun()


@st.dialog("新增数据集", width="medium")
def create_new_dataset(lang_opts, task_opts, tool_opts, org_opts, originality_opts):
    st.subheader("填写数据集信息")
    
    splits = st.session_state["add_ds_splits"]

    def add_split():
        splits.append({})
        st.session_state["add_ds_splits"] = splits

    def remove_split(idx):
        splits.pop(idx)
        st.session_state["add_ds_splits"] = splits

    split_cols = st.columns([2,8])
    with split_cols[0]:
        if st.button("➕ 添加切分", key="add_split_btn"):
            add_split()
            st.rerun()
    # 删除split按钮
    with split_cols[1]:
        btn_cols = st.columns(5)
        for idx, split in enumerate(splits):
            with btn_cols[idx]:
                if st.button(f"切分{idx+1} 🚫", key=f"del_split_{idx}"):
                    remove_split(idx)
                    st.rerun()
    with st.container(border= False, height= 500):
        with st.form("add_dataset_form", clear_on_submit=False):
            dataset_name = st.text_input("数据集名称", key="add_ds_name")
            dataset_version = st.text_input("数据集版本", key="add_ds_version")
            description = st.text_area("数据集描述", key="add_ds_desc")
            paper_url = st.text_input("论文链接", key="add_ds_paper")
            dataset_url = st.text_input("数据集链接", key="add_ds_url")
            github_url = st.text_input("GitHub 链接", key="add_ds_github")
            # 多选/可新建
            languages = st.multiselect("语言", options=lang_opts, key="add_ds_lang", accept_new_options = True)
            task_type = st.selectbox("任务类型", options=task_opts, key="add_ds_task", accept_new_options = True)
            tool_preference = st.multiselect("工具偏好", options=tool_opts, key="add_ds_tool", accept_new_options = True)
            num_total_samples = st.number_input("总样本数", min_value=0, step=1, key="add_ds_total")
            update_time = st.date_input("更新时间", key="add_ds_update")
            organization = st.selectbox("组织", options=org_opts, key="add_ds_org", accept_new_options = True)
            # dataset_splits 动态添加
            st.markdown("#### 切分（Splits）")
            for idx, split in enumerate(splits):
                with st.expander(f"切分 {idx+1}", expanded=False):
                    split["split_name"] = st.text_input(f"切分名", value=split.get("split_name", ""), key=f"split_name_{idx}")
                    split["display_name"] = st.text_input(f"显示名", value=split.get("display_name", ""), key=f"display_name_{idx}")
                    split["num_samples"] = st.number_input(f"样本数", min_value=0, step=1, value=split.get("num_samples", 0), key=f"num_samples_{idx}")
                    split["file_path"] = st.text_input(f"文件路径", value=split.get("file_path", ""), key=f"file_path_{idx}")
                    split["has_reference_answer"] = st.checkbox(f"有参考答案", value=split.get("has_reference_answer", False), key=f"has_ref_{idx}")
                    split["has_reasoning_process"] = st.checkbox(f"有推理过程", value=split.get("has_reasoning_process", False), key=f"has_reason_{idx}")
                    split["is_multimodal"] = st.checkbox(f"多模态", value=split.get("is_multimodal", False), key=f"is_multi_{idx}")
                    split["description"] = st.text_area(f"切分描述", value=split.get("description", ""), key=f"desc_{idx}")
                    # originality 多选
                    split["originality"] = st.multiselect(f"原创性", options=originality_opts, default=split.get("originality", []), key=f"orig_{idx}")
                    # column_mapping 字典
                    st.markdown("字段映射（可选）")
                    if "column_mapping" not in split:
                        split["column_mapping"] = {}
                    col_map = split["column_mapping"]
                    col_map["id"] = st.text_input("字段映射-id", value=col_map.get("id", ""), key=f"colmap_id_{idx}")
                    col_map["question"] = st.text_input("字段映射-question", value=col_map.get("question", ""), key=f"colmap_q_{idx}")
                    col_map["answer"] = st.text_input("字段映射-answer", value=col_map.get("answer", ""), key=f"colmap_a_{idx}")
                    col_map["reasoning_process"] = st.text_input("字段映射-reasoning_process", value=col_map.get("reasoning_process", ""), key=f"colmap_r_{idx}")
                    col_map["multimodal_file_path"] = st.text_input("字段映射-multimodal_file_path", value=col_map.get("multimodal_file_path", ""), key=f"colmap_m_{idx}")
                    col_map["reference"] = st.text_input("字段映射-reference", value=col_map.get("reference", ""), key=f"colmap_ref_{idx}")
                    split["column_mapping"] = col_map
                    # status 字典
                    st.markdown("状态（可选）")
                    if "status" not in split:
                        split["status"] = {"downloaded": False, "run_progess": {"processed_samples": 0, "correct_samples": 0, "correct_samples_path": None}}
                    status = split["status"]
                    status["downloaded"] = st.checkbox("已下载", value=status.get("downloaded", False), key=f"downloaded_{idx}")
                    run_progess = status.get("run_progess", {})
                    run_progess["processed_samples"] = st.number_input("已处理样本数", min_value=0, step=1, value=run_progess.get("processed_samples", 0), key=f"proc_{idx}")
                    run_progess["correct_samples"] = st.number_input("正确样本数", min_value=0, step=1, value=run_progess.get("correct_samples", 0), key=f"corr_{idx}")
                    run_progess["correct_samples_path"] = st.text_input("正确样本路径", value=run_progess.get("correct_samples_path", ""), key=f"corr_path_{idx}")
                    status["run_progess"] = run_progess
                    split["status"] = status
            #         # 删除split按钮
            #         if st.button(f"删除切分 {idx+1}", key=f"del_split_{idx}"):
            #             remove_split(idx)
            #             st.experimental_rerun()
            # if st.button("➕ 添加切分", key="add_split_btn"):
            #     add_split()
            #     st.rerun()

            submitted = st.form_submit_button("提交")
            if submitted:
                # 构造新数据集dict
                new_ds = {
                    "dataset_name": dataset_name,
                    "dataset_version": dataset_version,
                    "description": description,
                    "paper_url": paper_url,
                    "dataset_url": dataset_url,
                    "github_url": github_url,
                    "languages": languages,
                    "task_type": task_type,
                    "tool_preference": tool_preference,
                    "num_total_samples": num_total_samples,
                    "dataset_splits": [s for s in splits if s.get("split_name", "") != ""],
                    "update_time": update_time.strftime("%Y-%m-%d") if update_time else "",
                    "organization": organization,
                }
                # 更新本地datasets_summary并写入文件
                st.session_state["datasets_summary"].append(new_ds)
                try:
                    with open(DATASETS_JSON_PATH, "w", encoding="utf-8") as f:
                        import json
                        json.dump(st.session_state["datasets_summary"], f, ensure_ascii=False, indent=4)
                    # with st.spinner("填写成功，正在填入..."):  
                    st.toast("新增数据集成功, 已写入!", icon="💡")
                    time.sleep(1)
                except Exception as e:
                    st.error(f"写入本地文件失败：{e}")
                
                # 清空表单
                st.session_state["add_ds_form"] = {}
                st.session_state["add_ds_splits"] = [{},{},{}]
                time.sleep(1)
                st.rerun()


def initialze_sesstion_state():
    if "password_verified" not in st.session_state:
        st.session_state["password_verified"] = False
    # 初始化数据集
    if "datasets_summary" not in st.session_state:
        st.session_state["datasets_summary"] = []
        load_dataset(DATASETS_JSON_PATH)

    if "add_ds_splits" not in st.session_state:
        st.session_state["add_ds_splits"] = [{},{},{}]

    if "add_ds_form" not in st.session_state:
        st.session_state["add_ds_form"] = {}


def render_dataset_description_card(ds_row: pd.Series) -> None:
    # 使用 st.expander 结合 st.caption 展示数据集描述，并分行显示 description，同时展示每个 split 的 description
    dataset_name = ds_row['dataset_name']
    dataset_version = ds_row.get('dataset_version', '')
    description = ds_row.get('description', '无描述')
    paper_url = ds_row.get('paper_url') if len(str(ds_row.get('paper_url', ''))) else "无"
    dataset_url = ds_row.get('dataset_url') if len(str(ds_row.get('dataset_url', ''))) else "无"
    github_url = ds_row.get('github_url') if len(str(ds_row.get('github_url', ''))) else "无"
    splits = ds_row.get('dataset_splits', [])
    title_str = f"{dataset_name}（v{dataset_version}）" if dataset_version else dataset_name

    with st.expander(f"📝 数据集简介：{title_str}", expanded=True):
        # 分行显示 description
        for line in str(description).splitlines():
            st.caption(line if line.strip() else "　")
        st.caption(f"📄 论文: {paper_url}")
        st.caption(f"📦 数据集: {dataset_url}")
        st.caption(f"💻 GitHub: {github_url}")
        # 展示每个 split 的 description
        if splits and isinstance(splits, (list, tuple)):
            for idx, split in enumerate(splits):
                split_name = split.get('display_name') or split.get('split_name') or '未命名切分'
                split_desc = split.get('description', '无描述')
                st.caption(f"🔹 {dataset_name}/{split_name}", help=split_desc)

def render_all_datasets_markdown(ds_df):
    """
    生成所有数据集的 markdown 格式介绍
    """
    md_lines = []
    for _, ds_row in ds_df.iterrows():
        dataset_name = ds_row['dataset_name']
        dataset_version = ds_row.get('dataset_version', '')
        description = ds_row.get('description', '无描述')
        paper_url = ds_row.get('paper_url') if len(str(ds_row.get('paper_url', ''))) else "无"
        dataset_url = ds_row.get('dataset_url') if len(str(ds_row.get('dataset_url', ''))) else "无"
        github_url = ds_row.get('github_url') if len(str(ds_row.get('github_url', ''))) else "无"
        splits = ds_row.get('dataset_splits', [])

        title_str = f"# {dataset_name}（v{dataset_version}）" if dataset_version else f"# {dataset_name}"
        md_lines.append(title_str)
        md_lines.append("")
        # description
        md_lines.append(str(description))
        md_lines.append("")
        # 信息来源
        md_lines.append("## 信息来源")
        md_lines.append(f"* 论文: {paper_url}")
        md_lines.append(f"* 数据集: {dataset_url}")
        md_lines.append(f"* GitHub: {github_url}")
        md_lines.append("")
        # Split 说明
        md_lines.append("## Split 说明")
        if splits and isinstance(splits, (list, tuple)):
            for split in splits:
                split_name = split.get('display_name') or split.get('split_name') or '未命名切分'
                split_desc = split.get('description', '无描述')
                md_lines.append(f"* **{split_name}**: {split_desc}")
        else:
            md_lines.append("* 无切分信息")
        md_lines.append("\n---\n")
    return "\n".join(md_lines)


def page1() -> None:

    initialze_sesstion_state()

    col_title, col_select, col_add_delete = st.columns([7, 2, 1], vertical_alignment="bottom")
    with col_title:
        st.title("Deep Research 数据统计")
    with col_select:
        ds_df = to_dataset_rows(st.session_state["datasets_summary"])
        sp_df = to_split_rows(st.session_state["datasets_summary"])
        dataset_names = ds_df["dataset_name"].unique().tolist()
        selected_dataset = st.selectbox(
            "📚 选择数据集",
            options=["全部"] + sorted(dataset_names),
            index=0,
            key="dataset_selectbox",
            help="🌾 选择具体的数据集，能够看到数据集的相关介绍"
        )
    # st.write(sp_df)
    # 在标题和 selectbox 下方展示数据集描述卡片
    if selected_dataset != "全部":
        ds_row = ds_df[ds_df["dataset_name"] == selected_dataset].iloc[0]
        render_dataset_description_card(ds_row)
    else:
        # 用 st.expander 展示所有数据集简介
        with st.expander("📝 所有数据集简介", expanded=False):   
            for _, ds_row in ds_df.iterrows():
                dataset_name = ds_row['dataset_name']
                dataset_version = ds_row.get('dataset_version', '')
                description = ds_row.get('description', '无描述')
                paper_url = ds_row.get('paper_url') if len(str(ds_row.get('paper_url', ''))) else "无"
                dataset_url = ds_row.get('dataset_url') if len(str(ds_row.get('dataset_url', ''))) else "无"
                github_url = ds_row.get('github_url') if len(str(ds_row.get('github_url', ''))) else "无"
                splits = ds_row.get('dataset_splits', [])

                # 一级标题
                st.markdown(f"# {dataset_name}（v{dataset_version}）" if dataset_version else f"# {dataset_name}")
                # description
                st.markdown(description)
                # 信息来源
                st.markdown("## 信息来源")
                st.markdown(f"* 论文: {paper_url}")
                st.markdown(f"* 数据集: {dataset_url}")
                st.markdown(f"* GitHub: {github_url}")
                # Split 说明
                st.markdown("## Split 说明")
                if splits and isinstance(splits, (list, tuple)) and len(splits) > 0:
                    for split in splits:
                        split_name = split.get('display_name') or split.get('split_name') or '未命名切分'
                        split_desc = split.get('description', '无描述').replace('\n', '\n\t')
                        st.markdown(f"* **{split_name}**: \n\t{split_desc}")
                else:
                    st.markdown("* 无切分信息")
                st.markdown("---")

            # 下载按钮
            all_md = render_all_datasets_markdown(ds_df)
            st.download_button(
                label="📥 下载所有数据集简介（Markdown）",
                data=all_md,
                file_name="datasets_summary.md",
                mime="text/markdown"
            )

    # 侧边栏筛选器
    st.sidebar.header("数据筛选")
    lang_opts = get_unique_values(ds_df, "languages")
    task_opts = get_unique_values(ds_df, "task_type")
    tool_opts = get_unique_values(ds_df, "tool_preference")
    org_opts = get_unique_values(ds_df, "organization")
    originality_opts = get_unique_values(sp_df, "originality")

    # sel_langs = st.sidebar.multiselect("🌐 语言", options=lang_opts, default=[])
    sel_langs = []
    sel_tasks = st.sidebar.multiselect("📝 任务类型", options=task_opts, default=[])
    sel_tools = st.sidebar.multiselect("🛠️ 工具偏好", options=tool_opts, default=[])
    # sel_orgs = st.sidebar.multiselect("🏢 组织", options=org_opts, default=[])
    # sel_originality = st.sidebar.multiselect("✨ 原创性", options=originality_opts, default=[])
    # 组织和原创性作为隐藏筛选项，不在侧边栏显示，但依然参与后续筛选逻辑
    sel_orgs = []
    sel_originality = []

    tri_state = {"全部": None, "是": True, "否": False}
    sel_multimodal = tri_state[st.sidebar.selectbox("🖼️ 多模态", options=list(tri_state.keys()), index=0)]
    sel_has_ref = tri_state[st.sidebar.selectbox("📑 有参考答案", options=list(tri_state.keys()), index=0)]
    sel_has_reason = tri_state[st.sidebar.selectbox("🔎 有推理过程", options=list(tri_state.keys()), index=0)]

    view = st.sidebar.radio("👁️ 视图", options=["按数据集", "按切分"], index=0, horizontal=False)

    # ========== 新增数据集按钮与弹窗 ==========
    with col_add_delete:
        col_add, col_delete = st.columns([1,1], gap= None)
        with col_add:
            if st.button("➕"):
                create_new_dataset(lang_opts, task_opts, tool_opts, org_opts, originality_opts)
        with col_delete:
            if st.button("➖"):
                delete_datasets_dialog()
            

    # 应用筛选
    filtered_sp = apply_filters(
        sp_df,
        languages=sel_langs,
        task_types=sel_tasks,
        tools=sel_tools,
        organizations=sel_orgs,
        originality=sel_originality,
        is_multimodal=sel_multimodal,
        has_reference_answer=sel_has_ref,
        has_reasoning_process=sel_has_reason,
    )

    # 数据集视图：从切分筛选结果推导数据集
    filtered_ds_names = set(filtered_sp["dataset_name"].unique().tolist())
    filtered_ds = ds_df[ds_df["dataset_name"].isin(filtered_ds_names)].copy()

    # 如果选择了具体数据集，则只显示该数据集相关数据
    if selected_dataset != "全部":
        filtered_ds = filtered_ds[filtered_ds["dataset_name"] == selected_dataset]
        filtered_sp = filtered_sp[filtered_sp["dataset_name"] == selected_dataset]

    # 指标卡片
    render_metrics(filtered_ds, filtered_sp)

    # 图表与表格
    if view == "按数据集":
        render_dataset_charts(filtered_ds, filtered_sp, animate=True)
        st.subheader("📋 数据集明细")
        show_cols = [
            "dataset_name",
            "dataset_version",
            "organization",
            "task_type",
            "languages",
            "tool_preference",
            "num_total_samples",
            "update_time",
            "paper_url",
            "dataset_url",
            "github_url",
        ]
        st.dataframe(filtered_ds[show_cols], width='stretch', hide_index=True)
    else:
        render_split_charts(filtered_sp, animate=True)
        st.subheader("📋 切分明细")
        show_cols = [
            "dataset_name",
            "display_name",
            "num_samples",
            "is_multimodal",
            "has_reference_answer",
            "has_reasoning_process",
            "originality",
            "file_path",
            "processed_samples",
            "correct_samples",
            "correct_samples_path",
        ]
        st.dataframe(filtered_sp[show_cols], width='stretch', hide_index=True)
        render_progress(filtered_sp)

    # st.sidebar.markdown("---")
    # st.sidebar.caption("运行：`streamlit run streamlit_dashboard.py`")

    

    


if __name__ == "__main__":
    page1()
